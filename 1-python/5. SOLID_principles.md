# SOLID Principles
The SOLID principles are a set of five design principles that help software developers create more maintainable, scalable, and robust object-oriented systems. The acronym SOLID stands for:

## S - Single Responsibility Principle (SRP):

**Definition**: A class should have only one reason to change, meaning it should have only one job or responsibility.
Explanation: Each class should focus on a single part of the functionality provided by the software, and that responsibility should be entirely encapsulated by the class. By adhering to SRP, your code becomes easier to understand, test, and maintain.
**Example**: 
- *Bad example*
```python
class User:
    def __init__(self, username, password):
        self.username = username
        self.password = password

    def authenticate(self):
        # Logic for authenticating the user
        pass

    def save_to_database(self):
        # Logic to save user data to the database
        pass
```
- *Good example*
```python
class User:
    def __init__(self, username, password):
        self.username = username
        self.password = password

class UserAuthenticator:
    def authenticate(self, user):
        # Logic for authenticating the user
        pass

class UserRepository:
    def save_to_database(self, user):
        # Logic to save user data to the database
        pass
```
Here, the responsibilities are split into different classes: User, UserAuthenticator, and UserRepository.

# O - Open/Closed Principle (OCP):

**Definition**: Software entities (classes, modules, functions, etc.) should be open for extension but closed for modification.
**Explanation**: You should be able to add new functionality to a system without changing the existing code. This is typically achieved through polymorphism, where new subclasses can extend the behavior of existing classes.
**Example**:
- *Bad example*
```python
class PaymentProcessor:
    def process_payment(self, payment_type):
        if payment_type == "credit_card":
            self.process_credit_card()
        elif payment_type == "paypal":
            self.process_paypal()
    
    def process_credit_card(self):
        # Process credit card payment
        pass
    
    def process_paypal(self):
        # Process PayPal payment
        pass
```
This class requires modification whenever a new payment method is added, violating OCP.


- *Good example*
```python
class PaymentProcessor:
    def process_payment(self, payment_method):
        payment_method.process()

class CreditCardPayment:
    def process(self):
        # Process credit card payment
        pass

class PayPalPayment:
    def process(self):
        # Process PayPal payment
        pass

# Adding a new payment method
class BitcoinPayment:
    def process(self):
        # Process Bitcoin payment
        pass

```
Here, the PaymentProcessor class is open for extension but closed for modification. You can add new payment methods without changing existing code.


# L - Liskov Substitution Principle (LSP):

**Definition**: Subtypes must be substitutable for their base types without altering the correctness of the program.
**Explanation**: This principle ensures that derived classes can be used in place of their base classes without affecting the functionality. The subclass should extend the parent class's behavior in a logical and consistent way.
**Example**:

- *Bad example*
```python
class Bird:
    def fly(self):
        return "I can fly!"

class Penguin(Bird):
    def fly(self):
        raise NotImplementedError("Penguins can't fly")
```
A Penguin cannot be used as a substitute for a Bird without breaking the program, violating LSP.
- *Good example*
```python
class Bird:
    def lay_eggs(self):
        return "Laying eggs"

class FlyingBird(Bird):
    def fly(self):
        return "I can fly!"

class Penguin(Bird):
    def swim(self):
        return "I can swim!"
```
**Other example:**
- *Bad example*
Imagine you have a Rectangle class and a Square class. A square is a special type of rectangle where the width and height are always equal, so it might seem logical to inherit Square from Rectangle.

```python
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height

    def get_area(self):
        return self.width * self.height

class Square(Rectangle):
    def set_width(self, width):
        self.width = width
        self.height = width  # Ensure width and height are always equal

    def set_height(self, height):
        self.width = height  # Ensure width and height are always equal
        self.height = height
```
**Problem**: The Square class violates the Liskov Substitution Principle because it changes the behavior of the Rectangle class. Specifically, if you set the width of a Square, it also sets the height, which is unexpected behavior if you're working with a Rectangle object.

```python
def process_shape(rect: Rectangle):
    rect.set_width(10)
    rect.set_height(5)
    assert rect.get_area() == 50  # This will fail for Square

rect = Rectangle(2, 3)
process_shape(rect)  # Works fine

square = Square(2, 2)
process_shape(square)  # Fails because the area will be 25, not 50

```

- *Good example*
```python
class Shape:
    def get_area(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height

    def get_area(self):
        return self.width * self.height

class Square(Shape):
    def __init__(self, side_length):
        self.side_length = side_length

    def set_side_length(self, side_length):
        self.side_length = side_length

    def get_area(self):
        return self.side_length * self.side_length

```


# I - Interface Segregation Principle (ISP):

**Definition**: A client should not be forced to depend on interfaces it does not use.
**Explanation**: This principle advocates for creating specific, fine-grained interfaces rather than large, monolithic ones. Clients should only know about the methods that are relevant to them, making the system more flexible and easier to maintain.
**Example**: 
- *Bad example*
```python
class Vehicle:
    def drive(self):
        pass

    def fly(self):
        pass

class Car(Vehicle):
    def drive(self):
        # Car-specific driving logic
        pass

    def fly(self):
        raise NotImplementedError("Cars can't fly")
```
The Car class is forced to implement a fly method it doesn't need.

- Good example
```python
class Drivable:
    def drive(self):
        pass

class Flyable:
    def fly(self):
        pass

class Car(Drivable):
    def drive(self):
        # Car-specific driving logic
        pass

class Airplane(Drivable, Flyable):
    def drive(self):
        # Airplane-specific driving logic
        pass
    
    def fly(self):
        # Airplane-specific flying logic
        pass
```
Now, Car only implements the Drivable interface and not Flyable, adhering to ISP.


# D - Dependency Inversion Principle (DIP):

**Definition**: High-level modules should not depend on low-level modules. Both should depend on abstractions (e.g., interfaces or abstract classes). Additionally, abstractions should not depend on details; details should depend on abstractions.

**Explanation**: The idea is to reduce the coupling between high-level and low-level components by using abstractions. This makes the system more modular and easier to modify or extend.

**Example**: 
- *Bad example*
```python
class Database:
    def get_data(self):
        return "Data from the database"

class BusinessLogic:
    def __init__(self):
        self.database = Database()

    def process_data(self):
        data = self.database.get_data()
        return f"Processing {data}"
```
The BusinessLogic class depends directly on the Database class, violating DIP.

- *Good example*
```python
class DataSource:
    def get_data(self):
        pass

class Database(DataSource):
    def get_data(self):
        return "Data from the database"

class APIClient(DataSource):
    def get_data(self):
        return "Data from an API"

class BusinessLogic:
    def __init__(self, data_source: DataSource):
        self.data_source = data_source

    def process_data(self):
        data = self.data_source.get_data()
        return f"Processing {data}"
```
Here, BusinessLogic depends on the DataSource abstraction rather than a specific implementation, adhering to DIP. You can switch data sources without modifying the business logic.



**Summary**:
- **SRP** ensures that a class has only one responsibility.

- **OCP** ensures that systems can be extended without modifying existing code.

- **LSP** ensures that derived classes can be substituted for base classes without breaking the system.

- **ISP** ensures that clients are not forced to depend on interfaces they don't use.

- **DIP** reduces the dependency of high-level modules on low-level modules by relying on abstractions.