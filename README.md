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
- [x] 1. Arrays and Strings
- [x] 2. Linked Lists
- [x] 3. Stacks and Queues
- [x] 4. Trees and Graphs
- [x] 5. Bit Manipulation
- [x] 6. Math and Logic Puzzles
- [ ] 7. Object-Oriented Design (work in progress)
- [x] 8. Recursion and Dynamic Programming
- [x] 9. System Design and Scalability
- [x] 10. Sorting and Searching
- [x] 11. Testing
- [ ] 12. C and C++ (skipping for now)
- [ ] 13. Java (skipping for now)
- [x] 14. Databases
- [ ] 15. Threads and Locks
- [ ] 16. Moderate
- [ ] 17. Hard