# The Greeter class
class Greeter
  def initialize(name)
    @name = name.capitalize
  end
  def salute
    puts "Hello #{@name}!"
    #phi = (1 + Math.sqrt(5)) / 2
    phi = (1.0 + (5.0 ** (1.0/2.0))) / 2.0
		puts "O valor de Phi é: #{phi}"
    
  end
end

# Create a new object
g = Greeter.new("world")

# Output "Hello World!"
g.salute