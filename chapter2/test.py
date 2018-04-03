try:
    f = open("test.txt")
except FileNotFoundError:
    print("File is not found.")