import XCTest
import TensorFlow

@testable import Lookahead

fileprivate struct Network: Layer {
    var conv: Conv2D<Float>
    var dense: Dense<Float>

    init(){
        self.conv = Conv2D(filterShape: (1, 1, 3, 4), strides: (1,1), padding: .same, activation: relu)
        self.dense = Dense(inputSize: 4, outputSize: 2, activation: identity)
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
                                   upperBound: Tensor<Int32>(2))
        var previousLoss = Tensor<Float>(2.4)
        for ii in 1...24 {
            let (loss, grad) = valueWithGradient(at: model) {
                softmaxCrossEntropy(logits: $0(inputs), labels: labels)
            }
            print(optimizer.step, loss)
            XCTAssertLessThan(loss.scalarized(), previousLoss.scalarized(), "loss reduces for step \(optimizer.step)")
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
                                   upperBound: Tensor<Int32>(2))
        var previousLoss = Tensor<Float>(2.4)
        for ii in 1...24 {
            let (loss, grad) = valueWithGradient(at: model) {
                softmaxCrossEntropy(logits: $0(inputs), labels: labels)
            }
            print(optimizer.step, loss)
            XCTAssertLessThan(loss.scalarized(),
                              previousLoss.scalarized(),
                              "loss reduces for step \(optimizer.step)")
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

    func testKeyPathIterable() {
        var model = Network()
        var count = 0
        for _ in model.differentiableVectorView.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            count += 1
        }
        XCTAssertEqual(count, 4)
    }
    static var allTests = [
        ("testLookahead", testLookahead),
        ("testLookaheadFurther", testLookaheadFurther),
    ]
}
