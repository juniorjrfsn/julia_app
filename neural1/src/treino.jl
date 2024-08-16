import Pkg; Pkg.add("Flux")
using Flux
using Flux.Optimise: Adam
using MLDatasets

# Carregar o conjunto de dados MNIST
train_x, train_y = MNIST.traindata()
test_x, test_y = MNIST.testdata()

# Definir a arquitetura da rede neural
model = Chain(
    Dense(28*28, 128, relu),
    Dense(128, 10, softmax)
)

# Definir a função de perda
loss(x, y) = Flux.Losses.crossentropy(model(x), y)

# Definir o otimizador
opt = Adam()

# Treinar o modelo
for epoch in 1:10
    for (x,y) in zip(train_x, train_y)
        gs = gradient(() -> loss(x, y), params(model))
        Flux.Optimise.update!(opt, params(model), gs)
    end
end

# Avaliar o modelo
accuracy(x, y) = mean(argmax(model(x), dim=1) .== y)
println("Accuracy on test set: ", accuracy(test_x, test_y))