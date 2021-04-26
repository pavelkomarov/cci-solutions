# cci-solutions
My solutions to exercises in Cracking the Coding Interview, in Python.

I've named files like `cci_##.py`, where `##` is a chapter number. All answers for a chapter are written in a single file, with little `assert`s to test each one right after its implementation. Simply run the file to check all these tests.

Some helpful utilities I've had to build and use along the way:
- `from cci_02 import Node` for linked lists (singly-linked or doubly)
- `from cci_04 import TreeNode` for binary trees
- `from cci_10 import BitVector` for a bit map
- This [`pytest` example](https://github.com/pavelkomarov/projection-pursuit/blob/master/skpp/tests/test_skpp.py) demonstrates a complementary approach to chapter 11, which is focused much more on the high-level.
- `cci_14.py` for examples of how to use `sqlite3`
- `cci_15.py` for examples of how to use `threading`

Note that when importing all the code in the corresponding file gets run, because I haven't been careful enough to put things in `if __name__ == '__main__':` blocks.

| Chapter names | n_problems | Done | 
| --- | --- | --- |
| 1. Arrays and Strings | 9 | ✓ |
| 2. Linked Lists | 8 | ✓ |
| 3. Stacks and Queues |6 | ✓ |
| 4. Trees and Graphs | 12 | ✓ |
| 5. Bit Manipulation | 8 | ✓ |
| 6. Math and Logic Puzzles | 10 | ✓ |
| 7. Object-Oriented Design | 9 | 1 |
| 8. Recursion and Dynamic Programming | 14 | ✓ |
| 9. System Design and Scalability | 8 | ✓ |
| 10. Sorting and Searching | 11 | ✓ |
| 11. Testing | 6 | ✓ |
| 12. C and C++ | 11 | ✓ |
| 13. Java | 8 | 0 |
| 14. Databases | 7 | ✓ |
| 15. Threads and Locks | 7 | ✓ |
| 16. Moderate | 26 | ✓ |
| 17. Hard | 26 | 0 |
| | Σ 186 | 144 |
