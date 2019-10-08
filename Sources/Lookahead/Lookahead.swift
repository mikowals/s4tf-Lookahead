import TensorFlow

public protocol HasSlowWeights {
    associatedtype Model: Differentiable
    var slowWeights: Model.TangentVector {get set}
}

public class Lookahead<Opt: Optimizer, Model: Layer>: Optimizer & HasSlowWeights
    where Opt.Model == Model,
          Model.TangentVector.VectorSpaceScalar == Float,
          Opt.Scalar: TensorFlowFloatingPoint  {
    public typealias Model = Model
    public typealias Opt = Opt
    public var optimizer: Opt
    public var learningRate: Opt.Scalar {
        willSet { optimizer.learningRate = Opt.Scalar(newValue) }
    }
    public var step: Int = 0
    public var outerStep: Int = 6
    public var slowWeights: Model.TangentVector
    
    public init(for model: __shared Model, optimizer: Opt, outerStep: Int = 6){
        self.slowWeights = model.differentiableVectorView
        self.optimizer = optimizer
        self.learningRate = optimizer.learningRate
        self.outerStep = outerStep
    }
    
    public func update(_ model: inout Model, along direction: Model.TangentVector) {
        step += 1
        optimizer.update(&model, along: direction)
        if step % outerStep == 0 {
            var updateWeights = model.differentiableVectorView
            for kp in slowWeights.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
                let currentWeight = updateWeights[keyPath: kp]
                updateWeights[keyPath: kp] = (
                    (currentWeight + slowWeights[keyPath: kp]) / Float(2))  -  currentWeight
            }
            model.move(along: updateWeights)
            slowWeights = model.differentiableVectorView
        }
    }
}

public class LookaheadFurther<Opt: Optimizer, Model: Layer>: Lookahead<Opt, Model>
    where Opt: HasSlowWeights, Opt.Model == Model,
          Model.TangentVector.VectorSpaceScalar == Float,
          Opt.Scalar: TensorFlowFloatingPoint  {
    public override var slowWeights: Model.TangentVector {
        willSet { optimizer.slowWeights = newValue }
    }
}
