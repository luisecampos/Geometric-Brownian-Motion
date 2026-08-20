# Ask user for their name and greet them remove whitespace and capitalize first letter of each word
name = input("What's your name?").strip().title()

#Split user's name into first name and last name 
first, last = name.split(" ")

# Say hello to user
print(f"hello, {first}") # This will work because it is an f-string

print(f"hello, {last}")
"""
+, -, *, /, **, //, %
""" 

x = int(input("What's x?"))
y = int(input("What's y?"))

print(x + y)

