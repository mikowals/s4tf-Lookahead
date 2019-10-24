//
//  FOP.swift
//  
//
//  Created by Michael Kowalski on 24/10/19.
//

import TensorFlow

public class SGDFOP<Model: EuclideanDifferentiable>: Optimizer
    where Model.TangentVector: VectorProtocol & ElementaryFunctions & KeyPathIterable,
          Model.TangentVector.VectorSpaceScalar == Float {
    public typealias Model = Model
    /// The learning rate.
    public var learningRate: Float {
        willSet { optimizer.learningRate = newValue }
    }
    /// The momentum factor. It accelerates stochastic gradient descent in the relevant direction
    /// and dampens oscillations.
    public var momentum: Float
    /// The weight decay.
    public var decay: Float
    /// Use Nesterov momentum if true.
    public var nesterov: Bool
    /// The velocity state of the model.
    public var optimizer: SGD<Model>
    public var matrix: Model.TangentVector = .zero
    public var fopVelocity: Model.TangentVector = .zero
    var previousGrad: Model.TangentVector = .zero
    /// The set of steps taken.
    public var step: Int = 0

    public init(
        for model: __shared Model,
        optimizer: SGD<Model>,
        learningRate: Float = 0.01,
        momentum: Float = 0,
        decay: Float = 0,
        nesterov: Bool = false
    ) {
        precondition(learningRate >= 0, "Learning rate must be non-negative")
        precondition(momentum >= 0, "Momentum must be non-negative")
        precondition(decay >= 0, "Weight decay must be non-negative")

        self.optimizer = optimizer
        self.learningRate = learningRate
        self.momentum = momentum
        self.decay = decay
        self.nesterov = nesterov
        fopVelocity = model.differentiableVectorView
        previousGrad = fopVelocity
        matrix = fopVelocity
        for kp in matrix.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            previousGrad[keyPath: kp] = fopVelocity[keyPath: kp] * 0
            let t = matrix[keyPath: kp]
            if t.rank == 4 {
                matrix[keyPath: kp] = Raw.diag(diagonal: Tensor<Float>(ones: [t.shape[0] * t.shape[1]]))
                fopVelocity[keyPath: kp] = Tensor<Float>(zeros: [t.shape[0] * t.shape[1]])
            }
            else if t.rank == 2 {
                let std = rsqrt(Tensor<Float>(Float(t.shape[0]))) * 3
                matrix[keyPath: kp] = Tensor<Float>(randomNormal: [t.shape[0], t.shape[0]],         standardDeviation: std)
                fopVelocity[keyPath: kp] = Tensor<Float>(zeros: [t.shape[0], t.shape[0]])
            }
        }
    }

    public func update(_ model: inout Model, along direction: Model.TangentVector) {
        step += 1
        var fopDirection = direction
        let learningRate = self.learningRate * 1 / (1 + decay * Float(step))
        for kp in fopDirection.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            let dir = fopDirection[keyPath: kp]
            if dir.rank > 1 {
                let m = matrix[keyPath: kp]
                let pcm = matmul(m, transposed: false, m, transposed: true)
                var hyperGrad = Tensor<Float>(0)
                if dir.rank == 4 {
                    let prev = previousGrad[keyPath: kp].reshaped(to: [pcm.shape[1], dir.shape[2] * dir.shape[3]])
                    let spatialGrad = dir.reshaped(to: [pcm.shape[1], dir.shape[2] * dir.shape[3]])
                    fopDirection[keyPath: kp] = matmul(pcm, spatialGrad).reshaped(to: dir.shape)
                    hyperGrad = -matmul(spatialGrad, matmul(prev, transposed: true, m))
                    hyperGrad += matmul(matmul(prev, transposed: false, spatialGrad, transposed: true), m)
                }
                else if dir.rank == 2 {
                    let eye = Raw.diag(diagonal: Tensor<Float>(ones: [pcm.shape[0]]))
                    fopDirection[keyPath: kp] = matmul(eye + pcm, dir)
                    let prev = previousGrad[keyPath: kp]
                    hyperGrad = -matmul(dir, matmul(prev, transposed: true, m))
                    hyperGrad += matmul(matmul(prev, transposed: false, dir, transposed: true), m)
                }
                
                fopVelocity[keyPath: kp] = momentum * fopVelocity[keyPath: kp] - hyperGrad * learningRate
                matrix[keyPath: kp] += fopVelocity[keyPath: kp]
            }
        }
        optimizer.update(&model, along: fopDirection)
        previousGrad = direction
    }
}
