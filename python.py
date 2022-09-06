# Exercise 1
print('Hello world')

# Exercise 2


a = input('Input 1: ')
b = input('Input 2: ')

print('product of two intefer', float(a)*float(b))

# Exercise 3

import random;
print(random.randint(0,100))

# Exercise 4

for i in range(1, 101):
  print(i, end=" ")
  if(i%10==0):
      print("\n")
    
# Exercise 5

def function(a = float(input('Value a: ')), b = float(input('value b: '))): 
    if a>b:
        print("a is greater")
    elif b>a:
            print("b is grater")
    else:
                print("equal")
        
    

function()
     
# Exercise 6

import random

number= (random.randint(1,100))


userinput = float(input("User input: "))


if number>userinput:

   print("The answer is larger")

elif number<userinput:
      
    print("The answer is smaler")
    
# Exercise 7
import random

randomnum = random.randint(0,10)

for i in range(1, 6):
    print(randomnum, 'x' , i, '=', randomnum*i)
    


 