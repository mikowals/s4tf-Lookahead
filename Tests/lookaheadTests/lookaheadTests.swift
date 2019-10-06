import XCTest
@testable import lookahead

final class lookaheadTests: XCTestCase {
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(lookahead().text, "Hello, World!")
    }

    static var allTests = [
        ("testExample", testExample),
    ]
}
