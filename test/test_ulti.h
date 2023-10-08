#ifndef EASYNN_TETS_ULTI_H
#define EASYNN_TETS_ULTI_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <map>

#define QTEST_EXPECT(x, y, cond) \
    if (!((x)cond(y))) \
    { \
        printf("%s:%u: Failure\n", __FILE__, __LINE__); \
        if (strcmp(#cond, "==") == 0) \
        { \
            printf("Expected equality of these values:\n"); \
            printf("  %s\n", #x); \
            qtest_evaluate_if_required(#x, x); \
            printf("  %s\n", #y); \
            qtest_evaluate_if_required(#y, y); \
        } \
        else \
        { \
            printf("Expected: (%s) %s (%s), actual: %s vs %s\n", #x, #cond, #y, std::to_string(x).c_str(), std::to_string(y).c_str()); \
        } \
        *qtest_current_fail_cnt = *qtest_current_fail_cnt + 1; \
    }

#define EXPECT_EQ(x, y) QTEST_EXPECT(x, y, ==)
#define EXPECT_NE(x, y) QTEST_EXPECT(x, y, !=)
#define EXPECT_LT(x, y) QTEST_EXPECT(x, y, <)
#define EXPECT_LE(x, y) QTEST_EXPECT(x, y, <=)
#define EXPECT_GT(x, y) QTEST_EXPECT(x, y, >)
#define EXPECT_GE(x, y) QTEST_EXPECT(x, y, >=)

#define EXPECT_TRUE(x) \
    if (!((x))) \
    { \
        printf("%s:%u: Failure\n", __FILE__, __LINE__); \
        printf("Value of: %s\n", #x); \
        printf("  Actual: false\n"); \
        printf("Expected: true\n"); \
        *qtest_current_fail_cnt = *qtest_current_fail_cnt + 1; \
    }

#define EXPECT_FALSE(x) \
    if (((x))) \
    { \
        printf("%s:%u: Failure\n", __FILE__, __LINE__); \
        printf("Value of: %s\n", #x); \
        printf("  Actual: true\n"); \
        printf("Expected: false\n"); \
        *qtest_current_fail_cnt = *qtest_current_fail_cnt + 1; \
    }

template<typename T>
void qtest_evaluate_if_required(const char* str, T value)
{
    if (strcmp(str, std::to_string(value).c_str()) != 0)
    {
        std::cout << "    Which is: " << value << std::endl;
    }
}

#define ASSERT_EQ(x, y) \
    if ((x)!=(y)) \
    { \
        printf("%s:%u: Failure\n", __FILE__, __LINE__); \
        printf("Expected equality of these values:\n"); \
        printf("  %s\n", #x); \
        qtest_evaluate_if_required(#x, x); \
        printf("  %s\n", #y); \
        qtest_evaluate_if_required(#y, y); \
        *qtest_current_fail_cnt = *qtest_current_fail_cnt + 1; \
        return; \
    }

#define QTEST_ESCAPE_COLOR_RED "\x1b[31m"
#define QTEST_ESCAPE_COLOR_GREEN "\x1b[32m"
#define QTEST_ESCAPE_COLOR_YELLOW "\x1b[33m"
#define QTEST_ESCAPE_COLOR_END "\x1b[0m"

class TestEntity
{
private:
    TestEntity() { };
    ~TestEntity() { };

public:
    std::string make_proper_str(size_t num, const std::string str, bool uppercase = false)
    {
        std::string res;
        if (num > 1)
        {
            if (uppercase)
                res = std::to_string(num) + " " + str + "S";
            else
                res = std::to_string(num) + " " + str + "s";
        }
        else
        {
            res = std::to_string(num) + " " + str;
        }
        return res;
    }

public:
    TestEntity(const TestEntity& other) = delete;
    TestEntity operator=(const TestEntity& other) = delete;
    
    static TestEntity& get_instance()
    {
        static TestEntity entity;
        return entity;
    }
    int add(std::string test_set_name, std::string test_name, std::function<void(int*)> f, const char* fname)
    {
        TestItem item(f, fname);
        test_sets[test_set_name].test_items.emplace_back(item);
        return 0;
    }

    int set_filter(std::string _filter)
    {
        filter = _filter;
        return 0;
    }

    int run_all_test_functions()
    {
        std::map<std::string, TestSet>::iterator it = test_sets.begin();
        for (; it != test_sets.end(); it++)
        {
            std::string test_set_name = it->first;
            TestSet& test_set = it->second;
            std::vector<TestItem>& test_items = test_set.test_items;

            int cnt = 0;
            for (int i = 0; i < test_items.size(); i++)
            {
                const std::string fname = test_items[i].fname;
                if (filter.length() == 0 || (filter.length() > 0 && strmatch(fname, filter)))
                {
                    cnt++;
                }
            }

            if (cnt == 0) continue;

            matched_test_set_count++;

            const std::string test_item_str = make_proper_str(cnt, "test");
            printf("%s[----------]%s %s from %s\n", QTEST_ESCAPE_COLOR_GREEN, QTEST_ESCAPE_COLOR_END, test_item_str.c_str(), it->first.c_str());
            for (int i = 0; i < test_items.size(); i++)
            {
                auto f = test_items[i].f;
                std::string fname = test_items[i].fname;
                if (filter.length() > 0 && !strmatch(fname, filter))
                {
                    continue;
                }

                matched_test_case_count++;

                int qtest_current_fail_cnt = 0;
                printf("%s[ RUN      ]%s %s\n", QTEST_ESCAPE_COLOR_GREEN, QTEST_ESCAPE_COLOR_END, fname.c_str());
                f(&qtest_current_fail_cnt);
                if (qtest_current_fail_cnt == 0)
                {
                    printf("%s[       OK ]%s %s (0 ms)\n", QTEST_ESCAPE_COLOR_GREEN, QTEST_ESCAPE_COLOR_END, fname.c_str());
                }
                else
                {
                    printf("%s[  FAILED  ]%s %s (0 ms)\n", QTEST_ESCAPE_COLOR_RED, QTEST_ESCAPE_COLOR_END, fname.c_str());
                    qtest_fail_cnt++;
                }
                test_items[i].success = (qtest_current_fail_cnt == 0);
            }
            printf("%s[----------]%s %s from %s (0 ms total)\n", QTEST_ESCAPE_COLOR_GREEN, QTEST_ESCAPE_COLOR_END, test_item_str.c_str(), it->first.c_str());
            printf("\n");
        }

        printf("%s[----------]%s Global test environment tear-down\n", QTEST_ESCAPE_COLOR_GREEN, QTEST_ESCAPE_COLOR_END);
        std::string tests_str = make_proper_str(matched_test_case_count, "test");
        std::string suite_str = make_proper_str(matched_test_set_count, "test suite");
        printf("%s[==========]%s %s from %s ran. (0 ms total)\n",
            QTEST_ESCAPE_COLOR_GREEN, QTEST_ESCAPE_COLOR_END,
            tests_str.c_str(),
            suite_str.c_str()
        );

        int passed_test_count = matched_test_case_count - qtest_fail_cnt;
        std::string how_many_test_str = make_proper_str(passed_test_count, "test");
        printf("%s[  PASSED  ]%s %s.\n", QTEST_ESCAPE_COLOR_GREEN, QTEST_ESCAPE_COLOR_END, how_many_test_str.c_str());

        if (qtest_fail_cnt)
        {
            std::string failed_test_str = make_proper_str(qtest_fail_cnt, "test");
            printf("%s[  FAILED  ]%s %s, listed below:\n", QTEST_ESCAPE_COLOR_RED, QTEST_ESCAPE_COLOR_END, failed_test_str.c_str());

            std::map<std::string, TestSet>::iterator it = test_sets.begin();
            for (; it != test_sets.end(); it++)
            {
                std::string test_set_name = it->first;
                TestSet test_set = it->second;
                std::vector<TestItem> test_items = test_set.test_items;
                for (int i = 0; i < test_items.size(); i++)
                {
                    if (!test_items[i].success)
                    {
                        printf("%s[  FAILED  ]%s %s\n", QTEST_ESCAPE_COLOR_RED, QTEST_ESCAPE_COLOR_END, test_items[i].fname.c_str());
                    }
                }
            }
        }

        if (qtest_fail_cnt > 0)
        {
            std::string failed_test_str = make_proper_str(qtest_fail_cnt, "FAILED TEST", true);
            printf("\n %s\n", failed_test_str.c_str());
        }

        return 0;
    }

private:
    // https://leetcode.cn/problems/wildcard-matching/solutions/315802/tong-pei-fu-pi-pei-by-leetcode-solution/
    /// @param s string
    /// @param p pattern
    bool strmatch(std::string s, std::string p)
    {
        auto allStars = [](const std::string& str, int left, int right) {
            for (int i = left; i < right; ++i) {
                if (str[i] != '*') {
                    return false;
                }
            }
            return true;
        };
        auto charMatch = [](char u, char v)
        {
            return u == v || v == '?';
        };

        while (s.size() && p.size() && p.back() != '*')
        {
            if (charMatch(s.back(), p.back())) {
                s.pop_back();
                p.pop_back();
            }
            else {
                return false;
            }
        }
        if (p.empty()) {
            return s.empty();
        }

        int sIndex = 0;
        int pIndex = 0;
        int sRecord = -1;
        int pRecord = -1;
        while (sIndex < s.size() && pIndex < p.size()) {
            if (p[pIndex] == '*') {
                ++pIndex;
                sRecord = sIndex;
                pRecord = pIndex;
            }
            else if (charMatch(s[sIndex], p[pIndex])) {
                ++sIndex;
                ++pIndex;
            }
            else if (sRecord != -1 && sRecord + 1 < s.size()) {
                ++sRecord;
                sIndex = sRecord;
                pIndex = pRecord;
            }
            else {
                return false;
            }
        }
        return allStars(p, pIndex, p.size());
    }


public:
    struct TestItem
    {
        std::function<void(int*)> f;
        std::string fname;
        bool success;

        TestItem(std::function<void(int*)> _f, std::string _fname):
            f(_f), fname(_fname), success(true)
        {}
    };

    struct TestSet
    {
        std::vector<TestItem> test_items;
    };

    std::map<std::string, TestSet> test_sets;

public:
    int matched_test_case_count = 0;
    int matched_test_set_count = 0;

private:
    int qtest_fail_cnt = 0; // number of failures in one test set
    std::string filter;
};

#define TEST(set, name) \
    void qtest_##set##_##name(int* qtest_current_fail_cnt); \
    int qtest_mark_##set##_##name = TestEntity::get_instance().add(#set, #name, qtest_##set##_##name, #set "." #name); \
    void qtest_##set##_##name(int* qtest_current_fail_cnt) \

TEST(c, 1)
{
    EXPECT_EQ(1, 1);
}
TEST(c, 2)
{
    EXPECT_EQ(2, 3);
}

TEST(cd, 1)
{
    EXPECT_NE(2, 3);
}
TEST(cd, 2)
{
    EXPECT_NE(2, 2);
}

TEST(ce, 1)
{
    EXPECT_LE(2, 3);
}
TEST(ce, 2)
{
    EXPECT_LE(2, 2);
}

TEST(f, 1)
{
    EXPECT_LT(2, 5);
}

void InitQTest()
{
    int test_suite_count = TestEntity::get_instance().matched_test_set_count;
    int test_count = TestEntity::get_instance().matched_test_case_count;

    std::string test_suite_str = TestEntity::get_instance().make_proper_str(test_suite_count, "test suite");
    std::string test_count_str = TestEntity::get_instance().make_proper_str(test_count, "test");
    printf("%s[==========]%s Running %s from %s.\n", QTEST_ESCAPE_COLOR_GREEN, QTEST_ESCAPE_COLOR_END, test_count_str.c_str(), test_suite_str.c_str());
    printf("%s[----------]%s Global test environment set-up.\n", QTEST_ESCAPE_COLOR_GREEN, QTEST_ESCAPE_COLOR_END);
}

int RUN_ALL_TESTS()
{
    TestEntity::get_instance().run_all_test_functions();
    return 0;
}

#define QTEST_FILTER(filter_str) \
    TestEntity::get_instance().set_filter(filter_str)

int main()
{
    InitQTest();
    QTEST_FILTER("c*.1");
    //QTEST_FILTER("f?.1");
    return RUN_ALL_TESTS();
}


#endif