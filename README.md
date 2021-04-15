# cci-solutions
My solutions to exercises in Cracking the Coding Interview, in Python.

I've named files like `cci_#.py`, where `#` is a chapter number. All answers for a chapter are written in a single file, with little `assert`s to test each one right after its implementation. Simply run the file to check all these tests.

Some helpful utility classes I've had to build along the way:
- `from cci_02 import Node` for linked lists
- `from cci_04 import TreeNode` for binary trees
- `from cci_10 import BitVector` for a bit map
Note that these imports run all the code in the corresponding files, because I haven't been careful enough to put things in `if __name__ == '__main__':` blocks.

Chapter names:
1. Arrays and Strings
2. Linked Lists
3. Stacks and Queues
4. Trees and Graphs
5. Bit Manipulation
6. Math and Logic Puzzles
7. Object-Oriented Design [WIP]
8. Recursion and Dynamic Programming
9. System Design and Scalability
10. Sorting and Searching
11. Testing
12. C and C++ (skipping for now)
13. Java (skipping for now)
14. Databases
15. Threads and Locks
16. Moderate
17. Hard