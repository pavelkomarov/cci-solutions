# cci-solutions
My solutions to exercises in Cracking the Coding Interview, in Python.

I've named files like `cci_##.py`, where `##` is a chapter number. All answers for a chapter are written in a single file, with little `assert`s to test each one right after its implementation. Simply run the file to check all these tests.

Some helpful utilities I've had to build and use along the way:
- `from cci_02 import Node` for linked lists
- `from cci_04 import TreeNode` for binary trees
- `from cci_10 import BitVector` for a bit map
- `cci_14.py` for examples of how to use `sqlite3`

Note that when importing all the code in the corresponding file gets run, because I haven't been careful enough to put things in `if __name__ == '__main__':` blocks.

Chapter names:
1. [x] Arrays and Strings
2. [x] Linked Lists
3. [x] Stacks and Queues
4. [x] Trees and Graphs
5. [x] Bit Manipulation
6. [x] Math and Logic Puzzles
7. [ ] Object-Oriented Design (work in progress)
8. [x] Recursion and Dynamic Programming
9. [x] System Design and Scalability
10. [x] Sorting and Searching
11. [x] Testing
12. [ ] C and C++ (skipping for now)
13. [ ] Java (skipping for now)
14. [x] Databases
15. [ ] Threads and Locks
16. [ ] Moderate
17. [ ] Hard