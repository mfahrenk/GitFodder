# Here is some basic python code
# These examples are from https://developers.google.com/edu/python/introduction
# There are some minor edits to get it to work for Python 3.5


# import modules used here -- sys allows one to access terminal inputs
import sys

def main():
  print('Hello there', sys.argv[1])
  # Command line args are in sys.argv[1], .argv[2], ...
  # sys.argv[0] is the script name itself and can be ignored
  # the expected call is the something like Python MyCode.py Max
  # where                                                     ^ is sys.argv[1]
  # should return Hello there Max


# Define a repeat function that takes two inputs
  def repeat(s, exclaim):
    """
    -----------------------------------------------------------------------------------
    Return the string 's' repeated 3 times.
    If exclaim is true, add !
    """
	
    result = s + s + s # can also use s*3 which is faster
    if exclaim:
      result = result + '!'
    elif not exclaim:
        result = results + ' <- typo not picked up until run time (i.e. if exclaim is FALSE)' 
    else:
        result = 'I"ve got nothing for you'
    
    return result
    """
    -----------------------------------------------------------------------------------
    """	
  # print above function
  print(repeat('Yay',True))

  # Now some fun with strings
  
  pi = 3.14156	
  # text = 'The value of pi is ' + pi # need to convert pi to a string	
  text = 'The value of pi is ' + str(pi)
  print(text)
	
  # Each character in the sting is indexed, e.g.
  print(text[13] + text[5] + text[0])
  
  print(text[13:15] + ' is ' + text[-7:])
  
  # you can use prefixes like r which will make a print out everything in raw form (ignore
  # \n which normally means newline for instance
  raw = r'this\t\n and that'
  print(raw)
  print('this\t\n and that')
  
  # you can put together a sting using the %
  # %d = int, %s = string, %f or %g = float
  # the tuple on the right designates the entries
  text = ("%d  little pigs come out or I'll %s and %s and %s" % 
  (3, 'huff', 'puff', 'blow down')) # add parentheses for line continuation
  print(text)
  
  # lists in python work similar to char lists in MATLAB
  MyList = ['Max','is','learning','Python']
  print(MyList[2]) #learning
  
  #note that an assignment for list does not make a copy, it just makes a pointer
  PointToMyList = MyList
  print(PointToMyList[2]) #Learning
  MyList[2] = 'NotLearning'
  print(PointToMyList[2]) #NotLearning 
  
  #append two lists
  print(MyList + MyList)
  
  #you can use lists to iterate over specific cases
  squares = [1, 4, 9, 16] #n^2
  sum = 0
  for num in squares:
    sum += num
    
  print(sum) #30
  
  #Use *in* to test if a value is in a collection
  list = ['larry', 'curly', 'moe']
  if 'curly' in list:
    print('yay')  
	
  #here is a standard for loop
  for i in range(100): #0..99
    if i == 99: 
      print(i)
    
  #standard while loop
  i = 0
  while i < 10:
    i = i + 1
    
  print(i)
  
    
  #there are some common build in utilities for lists
  #list.append(elem) adds element to end of list
  #list.insert(index, elem) adds element to list at index, shifting other elements down
  #list.extend(list2) same as list += list2
  #list.index(elem) finds index of elem, throws ValueError if not found
  #list.remove(elem) finds and removes elem from list
  #list.sort() sorts list in place (sorted() is preferred)	
  #list.reverse()
  #list.pop(index) removes and returns the element at the given index 
  #list.pop() returns last element	
	
  #Tuples are often used to create a list that cannot have any one single element changed
  #The whole tuple can be over written though
  tuple = (1, 2, 'hi')
  print(len(tuple))
  print(tuple[2])   #hi
  #tuple[2] = 'bye'  #not allowed
  tuple = (1, 2, 'bye')
  tuple = ('hi',) #size-1 tuple, note the comma to distinguish the tuple from regular
                  #parentheses
                  
  #multiple assignment                
  (x, y, z) = (42, 13, 'hike')	
  print(z)
	
  #list comprehensions is a handy way to make a quick list of variables	
  squares = [ n*n for n in range(5) if n > 0] #range(5) = 0, 1, ..., 4
  #note that n^2 produces [3, 0, 1, 6] for some reason...?
  print(squares)
  
  #similarly for strings
  strs = ['hello', 'and', 'goodbye']
  shouting = [ s.upper() + '!' for s in strs ]
  print(shouting)
  
  #dictionaries - an efficient key/value hash table
  
	
# Standard boilerplate to call the main function to begin the program
# This is included in the main call. It is left out if you are making a module.
if __name__ == '__main__':
	main()