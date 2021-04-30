// compile with g++ cci_12.cc
// run with ./a.out, the default output name
#include <iostream> // so cout works
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
		a = *q; // is the value stored where q points
		*q = *p; // where q points gets the value stored where p points
		*p = a; // where p points gets value a
		p--; // p now points at the previous location
		q++; // q now points at the next location
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
 * & vs * is a challenge. Essentially * means we're taking a raw pointer by value, which holds an address,
 * and & means we're getting the address of something and then passing it in. The distinction is subtle and
 * really only has to do with whether you're handing object refs or pointers; in both cases you pass in an
 * address, which means we're effectively calling some object by reference.
 * https://stackoverflow.com/questions/5816719/difference-between-function-arguments-declared-with-and-in-c
 * https://stackoverflow.com/questions/114180/pointer-vs-reference
 */
Node* eight(Node* node, NodeMap& dict) { // Perform DFS over Nodes, creating nodes as we first visit
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

template <class T> class Nine { // smart pointer
	// I'm out of my depth enough with C++ that I more or less had to just look at the solutions and carefully
	// build up my understanding for this one. It showcases some language features.
	public:
		Nine(T* pointer) { // constructor given pointer to object
			std::cout << "in first constructor" << '\n';
			ref = pointer; // point to the same object as the pointer given
			ref_count = (unsigned*) malloc(sizeof(unsigned)); // malloc unsigned int, and return pointer to that location.
			*ref_count = 1; // increment the reference counter
		}

		Nine(Nine<T>& smart_pointer) { // constructor given another smart pointer
			std::cout << "in second constructor" << '\n';
			ref = smart_pointer.ref;
			ref_count = smart_pointer.ref_count;
			(*ref_count)++; // parens are important, because ++ binds more tightly
		}

		~Nine() {
			std::cout << "in destructor" << '\n';
			remove();
		}

		// "Override the equal operator, so that when you set one smart pionter equal to another, the old smart pointer
		// has its reference count decremented adn the new smart pointer has its reference count incremented."
		Nine<T>& operator=(Nine<T>& smart_pointer) {
			std::cout << "in assignment" << '\n';
			if (this == &smart_pointer) return *this; // I didn't realize this is a keyword in C++

			if (*ref_count > 0) remove();

			ref = smart_pointer.ref;
			ref_count = smart_pointer.ref_count;
			(*ref_count)++;
			return *this;
		}

	protected:
		T* ref; // the object this pointer points to
		unsigned* ref_count; // point to the reference count, so this object can increment/decrement it.

		void remove() {
			std::cout << "in remove" << '\n';
			(*ref_count)--;
			if (*ref_count == 0) {
				std::cout << "what a freeing feeling" << '\n';
				delete ref; // like the del keyword I'm more familiar with
				free(ref_count); // opposite of malloc
				ref = NULL; // "If the operand to the delete operator is a modifiable l-value, its value is undefined
				ref_count = NULL; // after the object is deleted." So set it explicitly so there's no undefined behavior.
			}
		}
};

void* ten_malloc(int size, int div) {
	// Idea: malloc a bigger chunk, and then return from that chunk. We'll need our size + up to div-1 extra
	// locations so we can find an address that's div-aligned within + room for a void pointer so we can store
	// where this block *really* begins, so we can free it properly
	void* p = malloc(size + div - 1 + sizeof(void*));
	if (p == NULL) return NULL; // happens if the system can't allocate the memory
	// To get the div-aligned location, we need (location) & ~(div - 1). div - 1 will give us log2(div) 1s on the right,
	// and ~(div-1) will flip that around, so we have a bitmask with 1s everywhere except for 0s at the last log2(div)
	// bits. That is, we chop off/waste at most div-1 bytes at the end of our malloced range. If we & directly with p,
	// we'll end up before p's location, so we have to take p forward by div-1 so q lands somewhere in [p,p+div-1]. We
	// also want that extra room for the extra void pointer, so q really lands in [p+sizeof(void*),p+sizeof(void*)+div-1].
	// We also have to do some casting on p, because the fact it's a void pointer means the system isn't sure how many
	// bytes to skip when we do arithmetic operations, and we actually just want to operate by single bytes.
	void* q = (void*) (((long) p + div - 1 + sizeof(void*)) & ~(div - 1));
	((void**) q)[-1] = p; // pretend q is a pointer to a void pointer rather than a void pointer itself, and now
	return q;			// we can go one void pointer's width backwards and store p there.
}

void ten_free(void* q) {
	std::cout << "I want to break free!\n";
	void* p = ((void**) q)[-1];
	free(p);
}

int** eleven(int n, int m) {
	int** arr = (int**) malloc(n*m*sizeof(int) + n*sizeof(int*)); // store n*m ints and n pointers to rows in a block
	if (arr == NULL) return NULL; // if system can't allocate the memory

	int* end_of_header = (int*) (arr + n); // Because arr's type is known, arithmetic like this actually offsets by
	for (int i = 0; i < n; i++) {			// sizeof*n bytes rather than just n bytes.
		arr[i] = end_of_header + i*m;
	}
	return arr;
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

	std::cout << "==nine==" << '\n';
	std::string* s1 = new std::string("hello"); // you have to use `new`, otherwise `delete` doesn't work.
	std::string* s2 = new std::string("lol wut");
	Nine<std::string> p1 (s1); // these two hit the first constructor
	Nine<std::string> p2 (s2);
	Nine<std::string> p3 (p1); // hits the second constructor
	p3 = p2; // hits the assignment, removing one ref and adding another
	p1 = p2; // hits assignement, and now ref_count for "hello" goes to zero, so we delete.
	std::cout << *s1 << '\n'; // prints just a newline, because s1 is deleted
	std::cout << *s2 << '\n'; // prints "lol wut", because s2 is still around
	// And then the destructor gets called a bunch as the program ends.
	// https://stackoverflow.com/questions/1962029/destructors-called-upon-program-termination

	std::cout << "==ten==" << '\n';
	void* q = ten_malloc(100, 16);
	std::cout << std::hex << (long) q << '\n';
	int m = ((long) q) % 16 == 0;
	std::cout << m << '\n';
	ten_free(q);

	std::cout << "==eleven==" << '\n';
	int** arr = eleven(4,5); // A double pointer behaves exactly like you'd expect an array to, but it. 
	for (int i = 0; i < 4; i++) {	// necessitates a header of secondary pointers. To save memory, you might
		for (int j = 0; j < 5; j++) {	// just keep n, m in some array class and calculate offset dynamically.
			arr[i][j] = i*10 + j;
			std::cout << std::dec << arr[i][j] << ',';
		}
		std::cout << '\n';
	}
	free(arr);

	std::cout << "\n\n\n";
}
