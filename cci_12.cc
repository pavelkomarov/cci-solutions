// compile with g++ cci_12.cc
// run with ./a.out, the default output name
#include <iostream> // so cout works
#include <cassert> // so I can write little tests
//using namespace std // allows you to not have to keep specifying std::thing

#include <fstream> // for file io and printf

void one(std::string fname, int K) {
	std::fstream f(fname);
	std::string lines[K];
	int i;

	while (f.peek() != EOF) { // peek, otherwise EOF line is included in last K
		getline(f, lines[i % K]);
		i++;
	}

	for (int j = 0; j < K; j++) {
		std::cout << lines[(i+j) % K] << '\n';
	}
}

void two(char* str) {
	char* p = str;
	/* Here I'm printing the literal values of the two pointers, which are the memory locations
	 * they point to. They're the same! Casting to long becomes necessary because &pointer gives
	 * you where the pointer is stored. *pointer gives you the value at the location it points
	 * to, and cout << pointer prints the pointed-to string up to the null character. Yet when
	 * I ++, >, ==, or whatever on pointers, I'm manipulating or comparing where they point to,
	 * which I frustratingly can't visualize via these three methods!
	 */
	//std::cout << std::hex << (long) str << '\n';
	//std::cout << std::hex << (long) p << '\n';
	while (*p != '\0') { // iterate a new pointer to the end of the string
		p++;
	}
	p--;
	char* q = str;
	char a;
	while (p > q) {
		a = *q;
		*q = *p;
		*p = a;
		p--;
		q++;
	}
}

/** Three
 * According to the docs https://www.cplusplus.com/reference/map/map/, the standard library map
 * is implemented as a binary search tree, navigated by key. By contrast, a hash table is backed
 * by an array and uses a hash function. This means insertion and lookup for the map are O(log n),
 * whereas for the hash table they're O(1). But in the hash table case, you have to deal with
 * collisions by chaining or pawing along for an empty location, and you have to deal with fillage
 * by copying to a sparser table from time to time, whereas in the map/bst there's no hash function
 * and no potential for collisions, and instead of copying we have to deal with rebalancing. For
 * small enough key,value sets, it may be better to skip allocating a big table and just use a map.
 */

/** Four
 * They're very similar to just normal functions in something like Java, but they're also kind of
 * like abstract functions. A virtual function with a definition can be overridden by a child
 * class, and then C++ will call the *child's* version of the method if an object is actually of
 * type child, no matter what that object is declared as. Whereas if the method is not virtual,
 * and the object is *declared* a parent but actually a child, the parent's version of the method
 * is bound to the object at compile time. The other usage is for virtual functions without a
 * definition. This seems to just serve as a note that the child class should override. The
 * mechanism underlying this is the "virtual table", or vtable, which gets attached to any class
 * with any virtual methods and stores addresses of versions of those methods that should be
 * called on a particular object. If C++ has to look in the vtable to find which method to call,
 * the method is resolved at runtime, and we call it "dynamic binding". For methods resolved at
 * compile time, we call it "static binding".
 */

/** Five
 * A shallow copy is a copy of an object where all its sub-object pointers refer to the original's
 * versions. A deep copy copies everything all the way down. I've only ever used deep copies,
 * because shallow ones tend to mea that modifying something over here has effects over there,
 * which I usually don't want and can become obscure bugs. The solutions suggest that a shallow
 * copy might be useful for passing information without data duplication. Seems like in that case
 * you might do just as well passing the original object. Gayle admits shallow copies are rarely
 * used. If all of an object's data is defined on the stack, rather than the heap, (all primatives?)
 * then shallow and deep copy are equivalent. "Note: C++ compiler implicitly creates a copy
 * constructor and overloads assignment operator in order to perform shallow copy at compile time."
 * https://www.geeksforgeeks.org/shallow-copy-and-deep-copy-in-c/
 */

/** Six
 * Volatile is used to denote that the value of a variable can change during run time due to
 * the actions of other threads, the OS, or sometimes the hardware. This can be important, so the
 * compiler knows not to assume C++ can used a cached value for or otherwise optimize over the
 * variable, that instead it must go look up the value every time.
 */

/** Seven
 * If the destructor in a base class isn't virtual, then it's possible an object gets statically
 * bound to the base class' destructor by the compiler. If such an object were destructed, all
 * memory-freeing operations done by the parent class would be completed, but we might still have
 * allocations for child-class-specific things that get completely missed.
 */

// Eight
#include <map> // a BST-based mapping template

struct Node {
	char val;
	Node* left;
	Node* right;
};
typedef std::map<Node*, Node*> NodeMap;

/**
 * & vs * is a challenge. Essentially * means we're taking a raw pointer, which holds an address, and &
 * means we're getting the address where something lives on the heap itself (calling by reference).
 * The distinction is subtle, but if you're not giving just x (calling by value), you have to do *x or &x.
 * https://stackoverflow.com/questions/5816719/difference-between-function-arguments-declared-with-and-in-c
 * https://stackoverflow.com/questions/114180/pointer-vs-reference
 */
Node* eight(Node* node, NodeMap& dict) { // Perform DFS over Nodes, creating nodes as we visit
	if (node == NULL) return NULL;
	std::cout << node->val << "\n"; // print visit order

	try { // map lookup throws exception if something not there
		return dict.at(node); // if we've already visited node, then return the copy we already made
	} catch (std::out_of_range) {
		Node* next = new Node; // The new copy of node
		dict[node] = next; // store in the map, so that if we visit node again, we can access the copy with a lookup
		next->val = node->val; // copy over data
		next->left = eight(node->left, dict); // left and right are given by the return of the recursion
		next->right = eight(node->right, dict);
		return next;
	}
}

int main() {
	std::cout << "==one==" << '\n';
	one("README.md", 10); // return as string and assert number of \n == 10?

	std::cout << "==two==" << '\n';
	char str[] = "hello\0"; // need to declare as array to make it mutable, otherwise bus error
	two(str); // str[] is actually a pointer to the beginning of the array
	std::cout << str << '\n';

	std::cout << "==eight==" << '\n';
	Node a = {'a', NULL, NULL}; //		      f <------.
	Node b = {'b', NULL, NULL}; //		    /   \      |
	Node c = {'c', &a, &b}; //			   c     e     |
	Node d = {'d', NULL, NULL}; //		 /	 \     \   |
	Node e = {'e', NULL, &d}; //		a     b     d  |
	Node f = {'f', &c, &e};	//			 \_____________!
	a.right = &f; // funny case where I have the actual object reference, not a pointer, so use .
				// This was confusing me, but I've landed on this explanation: A *reference* is basically
	NodeMap dict; // a name that C++ associates with some memory location where an object lives. A *pointer*
				// is a name that C++ associates to some memory location where a *pointer* lives.
	Node* copy = eight(&f, dict);
	assert(copy != &f && copy->val == 'f'); // not the same node and yet same value
	assert(copy->left != &c && copy->left->val == 'c');
	assert(copy->left->left != &a && copy->left->left->val == 'a');
	assert(copy->left->left->left == NULL);
	assert(copy->left->left->right == copy && copy->left->left->right->val == 'f');
	assert(copy->left->right != &b && copy->left->right->val == 'b');
	assert(copy->left->right->left == NULL);
	assert(copy->left->right->right == NULL);
	assert(copy->right != &e && copy->right->val == 'e');
	assert(copy->right->left == NULL);
	assert(copy->right->right != &d && copy->right->right->val == 'd');
	assert(copy->right->right->left == NULL);
	assert(copy->right->right->right == NULL);


}


