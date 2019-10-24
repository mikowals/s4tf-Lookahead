//
//  File.swift
//  
//
//  Created by Michael Kowalski on 9/10/19.
//
/*
import TensorFlow


/// Novograd optimizer.
///
/// Reference: ["Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks"](
/// https://arxiv.org/pdf/1905.11286.pdf)
public class Novograd<Model: Differentiable>: Optimizer
    where Model.TangentVector: VectorProtocol & PointwiseMultiplicative & ElementaryFunctions & KeyPathIterable,
          Model.TangentVector.VectorSpaceScalar == Float {
    public typealias Model = Model
    /// The learning rate.
    public var learningRate: Float
    /// A coefficient used to calculate the first and second moments of the gradients.
    public var beta1: Float
    /// A coefficient used to calculate the first and second moments of the gradients.
    public var beta2: Float
    /// A small scalar added to the denominator to improve numerical stability.
    public var epsilon: Float
    /// The learning rate decay.
    public var decay: Float
    /// The current step.
    public var step: Int = 0
    /// The first moments of the weights.
    public var firstMoments: Model.TangentVector
    /// The second moments of the weights.
    public var secondMoments: Model.TangentVector

    public init(
        for model: __shared Model,
        learningRate: Float = 1e-3,
        beta1: Float = 0.95,
        beta2: Float = 0.98,
        epsilon: Float = 1e-8,
        decay: Float = 0
    ) {
        precondition(learningRate >= 0, "Learning rate must be non-negative")
        precondition(0 <= beta1 && beta1 <= 1, "Beta parameter must be between 0 and 1")
        precondition(0 <= beta2 && beta2 <= 1, "Beta parameter must be between 0 and 1")
        precondition(decay >= 0, "Learning rate decay must be non-negative")

        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay = decay
        self.firstMoments = model.differentiableVectorView
        self.secondMoments = model.differentiableVectorView
    }

    public func update(_ model: inout Model, along direction: Model.TangentVector) {
        step += 1
        var scaledDirection = direction
        for kp in direction.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            let d = direction[keyPath: kp]
            let normSquared = d.squared().sum()
            //print(kp, normSquared)
            if step == 1 {
                secondMoments[keyPath: kp] = normSquared
                firstMoments[keyPath: kp] = d / (TensorFlow.sqrt(normSquared) + epsilon)
            } else {
                let v = secondMoments[keyPath: kp] * beta2 + normSquared * (1 - beta2)
                secondMoments[keyPath: kp] = v
                firstMoments[keyPath: kp] = firstMoments[keyPath: kp] * beta1 + d / (TensorFlow.sqrt(v) + epsilon)
            }
            scaledDirection[keyPath: kp] = -learningRate * firstMoments[keyPath: kp]
        }
        
        model.move(along: scaledDirection)
    }
}
*/
