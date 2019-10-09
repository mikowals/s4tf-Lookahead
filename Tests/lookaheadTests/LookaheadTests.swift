import XCTest
import TensorFlow

@testable import Lookahead

fileprivate struct Network: Layer {
    var conv: Conv2D<Float>
    var dense: Dense<Float>

    init(){
        self.conv = Conv2D(filterShape: (3, 3, 3, 16), strides: (1,1), padding: .same, activation: relu)
        self.dense = Dense(inputSize: 16, outputSize: 10, activation: identity)
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
        let sgd = SGD(for: model, learningRate: 0.1, momentum: 0.7)
        let optimizer = Lookahead(for: model,
                                  optimizer: sgd,
                                  outerStep: outerStep)
        
        XCTAssertEqual(optimizer.slowWeights, model.differentiableVectorView)
        let inputs = Tensor<Float>(randomNormal: [128, 32, 32, 3])
        let labels = Tensor<Int32>(randomUniform: [128],
                                   lowerBound: Tensor<Int32>(0),
                                   upperBound: Tensor<Int32>(10))
        var cummulativeUpdates: Network.TangentVector = .zero
        for ii in 1...2 {
            let grad = gradient(at: model) {
                softmaxCrossEntropy(logits: $0(inputs), labels: labels)
            }
            var oldSlowWeights = optimizer.slowWeights
            optimizer.update(&model, along: grad)
            cummulativeUpdates += sgd.velocity
            if ii % outerStep == 0 {
                XCTAssertEqual(optimizer.slowWeights, model.differentiableVectorView)
                for kp in oldSlowWeights.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
                    oldSlowWeights[keyPath: kp] += cummulativeUpdates[keyPath: kp] / 2.0
                    XCTAssertTrue(
                        optimizer.slowWeights[keyPath: kp]
                            .isAlmostEqual(to: oldSlowWeights[keyPath: kp], tolerance: Float(1e-6)),
                                  "outer loop matches estimate")
                }
                cummulativeUpdates = .zero
            } else {
                let estimate = optimizer.slowWeights + cummulativeUpdates
                let keyPaths = oldSlowWeights.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self)
                XCTAssertEqual(keyPaths.count, 4, "keypath finds all parameters")
                for kp in keyPaths {
                    XCTAssertTrue(model.differentiableVectorView[keyPath: kp].isAlmostEqual(to:
                        estimate[keyPath: kp], tolerance: Float(1e-6)),
                                  "inner loop matches SGD updates")
                }
                XCTAssertNotEqual(optimizer.slowWeights, model.differentiableVectorView)
            }
        }
        print(softmaxCrossEntropy(logits: model(inputs), labels: labels))
    }
    
    func testLookaheadFurther() {
        var model = Network()
        let outerStep1 = 6
        let outerStep2 = 36
        let sgd = SGD(for: model, learningRate: 0.1, momentum: 0.7)
        let l1 = Lookahead(for: model,
                                  optimizer: sgd,
                                  outerStep: outerStep1)
        let optimizer = LookaheadFurther(for: model, optimizer: l1, outerStep: outerStep2)
        
        XCTAssertEqual(optimizer.slowWeights, model.differentiableVectorView)
        let inputs = Tensor<Float>(randomNormal: [128, 32, 32, 3])
        let labels = Tensor<Int32>(randomUniform: [128],
                                   lowerBound: Tensor<Int32>(0),
                                   upperBound: Tensor<Int32>(10))
        for ii in 1...1000 {
            let grad = gradient(at: model) {
                softmaxCrossEntropy(logits: $0(inputs), labels: labels)
            }
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
        print(softmaxCrossEntropy(logits: model(inputs), labels: labels))
        XCTAssertEqual(optimizer.step, 1000)
        XCTAssertEqual(l1.step, 1000)
        XCTAssertEqual(sgd.step, 1000)
    }

    static var allTests = [
        ("testLookahead", testLookahead),
        ("testLookaheadFurther", testLookaheadFurther),
    ]
}
