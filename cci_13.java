
import java.util.*;
import java.util.stream.*;

/** One
 * Making a constructor private means *only* that class itself (and any inner classes) can instantiate it. So
 * if you want to use the class, you can't call it's constructor; you'll instead have to call some other public
 * method that the author has exposed, which then gets to govern all object creation. Or you'll have to use
 * one of the inner classes. Either way, the point is that the author gets complete control over how the class 
 * is instantiated. According to the internet, you might want to use this if you're pursuing a Singleton pattern.
 * I think Singletons are stupid, so this is overall a somewhat goofy language feature you probably don't need.
 */

class Two { // can't be public, or has to be named same thing as file. can't be private either.
	public static void main(String ... args) {
		System.out.println(method());
	}

	private static int method() {
		try {
			return 0;
			// throw new Exception(); unreachable statement
		} catch (Exception e) {
			System.out.println("catch");
		} finally {
			System.out.println("finally!");
		}
		return 1; // the compiler complains without this.
	}
}
// Running the above with `java Two` prints "finally" then the 0. So yes, it appears the finally block is called
// even in the case of early return. The solutions say it's possible a finally isn't executed if the thread is
// killed externally or the JVM dies. But I wouldn't expect recovery to be possible under those conditions, so
// it's like "The finally always executes unless a bomb drops on your program."

/** Three
 * `final` is a keyword that makes a variable unmodifyable or locks a pointer to point to whatever it's currently
 * pointing to. Apparently it can also be used with methods to make them un-overrideable or to classes to make
 * them un-subclassable.
 * `finally` is a block associated with try-catch blocks
 * According to the internet, `finalize()` is a method called by the garbage collector when no more references to
 * an object remain. You can override it if you want special deletion behavior, but I've always just used
 * java.lang.Object's default.
 */

/** Four
 * Templates and generics are similar, but how they work differs: Container<Type> in Java can work with any
 * particular Type, and effectively what happens is the compiler knows to do a bunch of casting to ensure that
 * things coming out of Container are marked as the right Type. Whereas in C++ Container<Type> ends up compiled
 * to something completely different for each specifc Type. So in Java, the type of Container<String> and
 * Container<Person> is the same, whereas they are different in C++. This has some consequences around static
 * variables: A Container<String> and Container<Person> will share the same static variables in Java, but not so
 * in C++. It's also the case that Java requires the Type to be an object, so you have to use Integer instead of
 * the primative int, for example, whereas C++ doesn't care. In Java you can restrict your generic to only work
 * with certain types (<T extends U>, meaning T is U or a subclass of U or an implementor of U), but in C++ you
 * can't restrict, which could lead to danger if you try to use a template with a class that doesn't have features
 * it expects.
 */

/** Five
 * They're all maps with the same interface. They only differ in how they're implemented. A HashMap is typically
 * an array of linked lists, offering amortized O(1) lookup and retrieval time. A TreeMap stores things in a BST.
 * The hints say it's a red-black tree in Java, which is self-balancing and offers amortized O(log n) lookup time.
 * According to the internet, a LinkedHashMap is like an ordinary HashMap, except it maintains a doubly-linked
 * list through all its elements, so that you can iterate them in the order they were added. Double linkages
 * ensure things can be removed in O(1). An interesting feature of the TreeMap is that you can iterate the keys
 * in order (alphabetical, or whatever comparitor you like) by performing a tree traversal. So if you need
 * particular ordering, you might want to use one of these alernatives to the HashMap.
 */

/** Six
 * Reflection is a bunch of properties and methods that allow you to see the class, methods, fields, etc. of an
 * object and get, set, invoke, etc. them programmatically by name or reference. Some times you want to be able to
 * interrogate and call this information, for example I used reflection heavily when writing my scikit-learn model
 * params -> xml function.
 */

/*numbers.forEach( (n) -> { System.out.println(n); } );* Seven **/
class Country {
	public String continent;
	public int population;

	public Country(String continent, int population) {
		this.continent = continent;
		this.population = population;
	}
}

class Seven { // everything has to be a class in Java
	public static void main(String ... args) {
		Country a = new Country("South America", 1000000);
		Country b = new Country("South America", 100000);
		Country c = new Country("South America", 250000);
		Country d = new Country("Asia", 1000000000);
		Country e = new Country("Africa", 250000000);

		List<Country> countries = Arrays.asList(a, b, c, d, e);

		// You'd use lambdas as arguments to filter, map, and reduce here. You can't just apply those to any
		// iterable like Python, so you have to pull out the Stream first (dumb)
		Stream<Integer> pops = countries.stream().map(
			x -> x.continent.equals("South America") ? x.population : 0);

		System.out.println(pops.reduce(0, (x, y) -> x + y)); // prints 1350000
	}
}

class Eight {
	private static Random rand = new Random(); // It makes no sense that *everything* has to be an object in java.
							// I miss calling simple libraries for this sort of thing.

	public static void main(String ... args) {
		List<Integer> list = Arrays.asList(1,2,3,4,5,6,7,8,9,10);

		HashMap<List<Integer>, Integer> n_occurrences = new HashMap<>();
		for (int i = 0; i < 100000; i++) {
			List<Integer> s = subset(list);
			n_occurrences.put(s, n_occurrences.getOrDefault(s, 0) + 1);
		}
		System.out.println(n_occurrences.values()); // each should occur about 100000/1024 times = 97.65625
		System.out.println("size = " + n_occurrences.size()); // should be 1024, because size of power set is 2^n
	}

	public static List<Integer> subset(List<Integer> list) {
		// coin flip for including each; just *why* with the Collector? Ugh
		return list.stream().filter(x -> rand.nextBoolean()).collect(Collectors.toList());
	}
}
