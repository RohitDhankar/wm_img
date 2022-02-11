
## https://www.youtube.com/watch?v=MBbVq_FIYDA
# super() = Function used to give access to the methods of a parent class.
#           Returns a temporary object of a parent class when used

class Rectangle_parentClass:
    def __init__(self, length, width):#,height): ## cant have HRIGHT here Not Ok 
        self.length = length
        self.width = width
        #DHANKAR Code Added below -- height
        #self.height = height ## cant have HRIGHT here Not Ok 

class Square_childClass(Rectangle_parentClass):
    def __init__(self, length, width):
        super().__init__(length,width)#,height) ## cant have HRIGHT here Not Ok 

    def area(self):
        return self.length*self.width

class Cube_childClass(Rectangle_parentClass):
    def __init__(self, length, width, height):
        super().__init__(length,width)#,height)
        self.height = height

    def volume(self):
        return self.length*self.width*self.height


square = Square_childClass(3, 3)
cube = Cube_childClass(3, 3, 3)

print(square.area())
print(cube.volume())