/**
 * @file test_main.cpp
 * @brief Minimal test harness — no external dependencies required.
 *
 * A lightweight test framework that mimics Google Test's interface
 * but compiles standalone. Each test is a function that uses
 * ASSERT/EXPECT macros to verify behavior.
 */

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <cstdlib>

// ─── Test Registration ───────────────────────────────────────

struct TestCase {
    std::string name;
    std::function<bool()> func;
};

static std::vector<TestCase>& get_tests() {
    static std::vector<TestCase> tests;
    return tests;
}

struct TestRegistrar {
    TestRegistrar(const char* name, std::function<bool()> func) {
        get_tests().push_back({name, std::move(func)});
    }
};

#define TEST(suite, name)                                         \
    [[maybe_unused]] static bool test_##suite##_##name();         \
    [[maybe_unused]] static TestRegistrar reg_##suite##_##name(   \
        #suite "::" #name, test_##suite##_##name);                \
    static bool test_##suite##_##name()

// ─── Assertion Macros ────────────────────────────────────────

#define ASSERT_TRUE(expr)                                         \
    do {                                                          \
        if (!(expr)) {                                            \
            std::cerr << "  FAIL: " << #expr                     \
                      << " [" << __FILE__ << ":" << __LINE__     \
                      << "]" << std::endl;                       \
            return false;                                         \
        }                                                         \
    } while(0)

#define ASSERT_FALSE(expr) ASSERT_TRUE(!(expr))

#define ASSERT_EQ(a, b)                                           \
    do {                                                          \
        if ((a) != (b)) {                                         \
            std::cerr << "  FAIL: " << #a << " == " << #b       \
                      << " (" << (a) << " != " << (b) << ")"    \
                      << " [" << __FILE__ << ":" << __LINE__     \
                      << "]" << std::endl;                       \
            return false;                                         \
        }                                                         \
    } while(0)

#define ASSERT_NE(a, b)                                           \
    do {                                                          \
        if ((a) == (b)) {                                         \
            std::cerr << "  FAIL: " << #a << " != " << #b       \
                      << " (" << (a) << " == " << (b) << ")"    \
                      << " [" << __FILE__ << ":" << __LINE__     \
                      << "]" << std::endl;                       \
            return false;                                         \
        }                                                         \
    } while(0)

#define ASSERT_GT(a, b)                                           \
    do {                                                          \
        if (!((a) > (b))) {                                       \
            std::cerr << "  FAIL: " << #a << " > " << #b        \
                      << " (" << (a) << " <= " << (b) << ")"    \
                      << " [" << __FILE__ << ":" << __LINE__     \
                      << "]" << std::endl;                       \
            return false;                                         \
        }                                                         \
    } while(0)

#define ASSERT_LT(a, b)                                           \
    do {                                                          \
        if (!((a) < (b))) {                                       \
            std::cerr << "  FAIL: " << #a << " < " << #b        \
                      << " (" << (a) << " >= " << (b) << ")"    \
                      << " [" << __FILE__ << ":" << __LINE__     \
                      << "]" << std::endl;                       \
            return false;                                         \
        }                                                         \
    } while(0)

#define ASSERT_GE(a, b)                                           \
    do {                                                          \
        if (!((a) >= (b))) {                                      \
            std::cerr << "  FAIL: " << #a << " >= " << #b       \
                      << " (" << (a) << " < " << (b) << ")"     \
                      << " [" << __FILE__ << ":" << __LINE__     \
                      << "]" << std::endl;                       \
            return false;                                         \
        }                                                         \
    } while(0)

// ─── Main Runner ─────────────────────────────────────────────

int main() {
    int passed = 0, failed = 0;
    auto& tests = get_tests();

    std::cout << "\n=== HFT Engine Test Suite ===" << std::endl;
    std::cout << "Running " << tests.size() << " tests...\n" << std::endl;

    for (auto& tc : tests) {
        std::cout << "[ RUN  ] " << tc.name << std::endl;
        bool ok = false;
        try {
            ok = tc.func();
        } catch (const std::exception& e) {
            std::cerr << "  EXCEPTION: " << e.what() << std::endl;
            ok = false;
        }

        if (ok) {
            std::cout << "[ PASS ] " << tc.name << std::endl;
            ++passed;
        } else {
            std::cout << "[ FAIL ] " << tc.name << std::endl;
            ++failed;
        }
    }

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Passed: " << passed << "/" << tests.size() << std::endl;
    if (failed > 0) {
        std::cout << "Failed: " << failed << std::endl;
    }
    std::cout << std::endl;

    return (failed > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
