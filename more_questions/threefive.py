"""Sum all the numbers divisible by 3 or 5 which are less than or equal to n.
"""

def func1(n):
    s = 0
    for i in range(n+1):
        if i % 3 == 0 or i % 5 == 0:
            s += i

    return(s)

# O(1) solution: Use Euler's trick
def func2(n):
    threes = n//3
    fives = n//5
    both = n//15

    return 3*threes*(threes+1)//2 + 5*fives*(fives+1)//2 - 15*both*(both+1)//2


print(func1(10), func2(10))
print(func1(20), func2(20))
print(func1(100), func2(100))        
