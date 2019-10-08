import XCTest
import TensorFlow

@testable import Lookahead

fileprivate struct MyEuclideanDifferentiableConv2D: MyEuclideanDifferentiable {
    var conv: Conv2D<Float>
    public var differentiableVectorView: TangentVector {
        get { TangentVector(conv: conv.differentiableVectorView) }
        set { conv.filter = newValue.conv.filter; conv.bias = newValue.conv.bias }
    }
    
    init(filterShape: (Int, Int, Int, Int)){
        self.conv = Conv2D(filterShape: filterShape,
                            strides: (1,1),
                            padding: .same,
                            activation: relu )
    }
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        conv(input)
    }
}

fileprivate struct MyEuclideanDifferentiableDense: MyEuclideanDifferentiable {
    var dense: Dense<Float>
    public var differentiableVectorView: TangentVector {
        get { TangentVector(dense: dense.differentiableVectorView) }
        set { dense.weight = newValue.dense.weight; dense.bias = newValue.dense.bias }
    }
    
    init(inputSize: Int, outputSize: Int) {
        self.dense = Dense(inputSize: inputSize, outputSize: outputSize)
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        dense(input)
    }
}

fileprivate struct Network: Layer & MyEuclideanDifferentiable {
    var conv: MyEuclideanDifferentiableConv2D
    var dense: MyEuclideanDifferentiableDense
    var differentiableVectorView: TangentVector {
        get { TangentVector(conv: conv.differentiableVectorView, dense: dense.differentiableVectorView) }
        set { conv.differentiableVectorView = newValue.conv
              dense.differentiableVectorView = newValue.dense }
    }
    init(){
        self.conv = MyEuclideanDifferentiableConv2D(filterShape: (3, 3, 3, 16))
        self.dense = MyEuclideanDifferentiableDense(inputSize: 16, outputSize: 10)
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return dense(conv(input).mean(squeezingAxes: [1, 2]))
    }
}

final class LookaheadTests: XCTestCase {
    func testLookahead() {
        var model = Network()
        let outerStep = 6
        var optimizer = Lookahead(for: model,
                                  optimizer: SGD(for: model, learningRate: 0.1, momentum: 0.7),
                                  outerStep: outerStep)
        
        XCTAssertEqual(optimizer.slowWeights, model.differentiableVectorView)
        let inputs = Tensor<Float>(randomNormal: [128, 32, 32, 3])
        let labels = Tensor<Int32>(randomUniform: [128],
                                   lowerBound: Tensor<Int32>(0),
                                   upperBound: Tensor<Int32>(10))
        var previousLoss = Tensor<Float>(2.4)
        for ii in 1...24 {
            let (loss, grad) = valueWithGradient(at: model) {
                softmaxCrossEntropy(logits: $0(inputs), labels: labels)
            }
            XCTAssertLessThan(loss.scalarized(), previousLoss.scalarized())
            previousLoss = loss
            optimizer.update(&model, along: grad)
            if ii % outerStep == 0 {
                XCTAssertEqual(optimizer.slowWeights, model.differentiableVectorView)
            } else {
                XCTAssertNotEqual(optimizer.slowWeights, model.differentiableVectorView)
            }
        }
    }
    
    func testLookaheadFurther() {
        var model = Network()
        let outerStep1 = 6
        let outerStep2 = 12
        var sgd = SGD(for: model, learningRate: 0.1, momentum: 0.7)
        var l1 = Lookahead(for: model,
                                  optimizer: sgd,
                                  outerStep: outerStep1)
        let optimizer = LookaheadFurther(for: model, optimizer: l1, outerStep: outerStep2)
        
        XCTAssertEqual(optimizer.slowWeights, model.differentiableVectorView)
        let inputs = Tensor<Float>(randomNormal: [128, 32, 32, 3])
        let labels = Tensor<Int32>(randomUniform: [128],
                                   lowerBound: Tensor<Int32>(0),
                                   upperBound: Tensor<Int32>(10))
        var previousLoss = Tensor<Float>(2.4)
        for ii in 1...24 {
            let (loss, grad) = valueWithGradient(at: model) {
                softmaxCrossEntropy(logits: $0(inputs), labels: labels)
            }
            XCTAssertLessThan(loss.scalarized(),
                              previousLoss.scalarized(),
                              "loss reduces with each training step")
            previousLoss = loss
            optimizer.update(&model, along: grad)
            if ii % outerStep1 == 0 {
                XCTAssertEqual(l1.slowWeights,
                               model.differentiableVectorView,
                               "model weights match inner slowWeights")
                if ii % outerStep2 > 0 {
                    XCTAssertNotEqual(l1.slowWeights,
                                      optimizer.slowWeights,
                                      "inner slowWeights update independent of outer slowWeights")
                }
            } else {
                XCTAssertNotEqual(optimizer.slowWeights,
                                  model.differentiableVectorView,
                                  "model updates without outer slowWeights")
                XCTAssertNotEqual(l1.slowWeights,
                                  model.differentiableVectorView,
                                  "model updates without inner slowWeights")
                
            }
            
            if ii % outerStep2 == 0 {
                XCTAssertEqual(optimizer.slowWeights,
                               model.differentiableVectorView,
                               "outer slowWeights match model")
                XCTAssertEqual(optimizer.slowWeights,
                               l1.slowWeights,
                               "inner and outer slowWeights updated together")
            }
        }
        XCTAssertEqual(optimizer.step, 24)
        XCTAssertEqual(l1.step, 24)
        XCTAssertEqual(sgd.step, 24)
    }

    static var allTests = [
        ("testLookahead", testLookahead),
        ("testLookaheadFurther", testLookaheadFurther),
    ]
}
