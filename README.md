# julia_app

## Create app
```
$ cd julia_app
julia_app $
$ julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.8.5 (2023-01-08)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia>
julia > ]
(@v1.8) pkg> generate calculos
(@v1.8) pkg> generate financ
(@v1.8) pkg> generate objetos
(@v1.8) pkg> generate servidor
(@v1.8) pkg> generate neural1
(@v1.8) pkg> generate perceptronxor

```

## adicionar dependencias
```
julia> ]
(@v1.8) pkg> add JSON
(@v1.8) pkg> add Pkg
(@v1.8) pkg> add SQLite
(@v1.8) pkg> add DataFrames
(@v1.8) pkg> add Flux
(@v1.8) pkg> add StatsBase
(@v1.8) pkg> add CSV
(@v1.8) pkg> add MLJ
(@v1.8) pkg> add MLJFlux
(@v1.8) pkg> add Images
(@v1.8) pkg> add FileIO
(@v1.8) pkg> add Serialization
(@v1.8) pkg> add Distributions
(@v1.8) pkg> add ImageDraw
(@v1.8) pkg> add ImageCore
(@v1.8) pkg> add JSON3
(@v1.8) pkg> add Luxor
```


## **Executar o arquivo**
```
julia> ]
pkg> activate ./calculos
julia > import calculos;
julia > calculos.calcular();
julia > calculos.greet();
```
