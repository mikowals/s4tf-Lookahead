import XCTest

import lookaheadTests

var tests = [XCTestCaseEntry]()
tests += lookaheadTests.allTests()
XCTMain(tests)
