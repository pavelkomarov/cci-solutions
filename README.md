# cci-solutions
My solutions to exercises in Cracking the Coding Interview, in Python.

I've named files like `cci_##.py`, where `##` is a chapter number. All answers for a chapter are written in a single file, with little `assert`s to test each one right after its implementation. Simply run the file to check all these tests.

Some helpful utilities I've had to build and use along the way:
- `from cci_02 import Node` for linked lists
- `from cci_04 import TreeNode` for binary trees
- `from cci_10 import BitVector` for a bit map
- This [`pytest` example](https://github.com/pavelkomarov/projection-pursuit/blob/master/skpp/tests/test_skpp.py) demonstrates a complementary approach to chapter 11, which is focused much more on the high-level.
- `cci_14.py` for examples of how to use `sqlite3`
- `cci_15.py` for examples of how to use `threading`

Note that when importing all the code in the corresponding file gets run, because I haven't been careful enough to put things in `if __name__ == '__main__':` blocks.

| Done | Chapter names | n_completed | n_problems | 
| --- | --- | --- | --- |
| [x] | 1. Arrays and Strings | 9 | 9 |
| [x] | 2. Linked Lists | 8 | 8 |
| [x] | 3. Stacks and Queues |6 | 6 |
| [x] | 4. Trees and Graphs | 12 | 12
| [x] | 5. Bit Manipulation | 8 | 8 |
| [x] | 6. Math and Logic Puzzles | 10 | 10 |
| [ ] | 7. Object-Oriented Design | 1 | 9 |
| [x] | 8. Recursion and Dynamic Programming | 14 | 14 |
| [x] | 9. System Design and Scalability | 8 | 8 |
| [x] | 10. Sorting and Searching | 11 | 11 |
| [x] | 11. Testing | 6 | 6 |
| [ ] | 12. C and C++ | 0 | 11 |
| [ ] | 13. Java | 0 | 8 |
| [x] | 14. Databases | 7 | 7 |
| [x] | 15. Threads and Locks | 7 | 7 |
| [ ] | 16. Moderate | 11 | 26 |
| [ ] | 17. Hard | 0 | 26 |
| | | Σ 118 | 186 |
