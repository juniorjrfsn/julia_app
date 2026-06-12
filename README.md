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
(@v1.8) pkg> generate mercadoRNA
(@v1.8) pkg> generate lstmrnntrain
(@v1.8) pkg> generate lstmcnntrain
(@v1.8) pkg> generate cnncheckin
(@v1.8) pkg> generate z_hrml
(@v1.8) pkg> generate autonomo
(@v1.8) pkg> generate webcamcnn
(@v1.8) pkg> generate cnn
(@v1.8) pkg> generate webcamcnnwindows
(@v1.8) pkg> generate chatbot
(@v1.8) pkg> generate webapp

```

## adicionar dependencias

```
julia> ]
(@v1.12) pkg> add HTTP
(@v1.12) pkg> add JSON3
(@v1.12) pkg> add JSON
(@v1.12) pkg> add Pkg
(@v1.12) pkg> add SQLite
(@v1.12) pkg> add DataFrames
(@v1.12) pkg> add Flux
(@v1.12) pkg> add StatsBase
(@v1.12) pkg> add CSV
(@v1.12) pkg> add MLJ
(@v1.12) pkg> add MLJFlux
(@v1.12) pkg> add Images
(@v1.12) pkg> add FileIO
(@v1.12) pkg> add Serialization
(@v1.12) pkg> add Distributions
(@v1.12) pkg> add ImageDraw
(@v1.12) pkg> add ImageCore
(@v1.12) pkg> add JSON3
(@v1.12) pkg> add Luxor
(@v1.12) pkg> add ImageMagick
(@v1.12) pkg> add QuartzImageIO
(@v1.12) pkg> add JLD2
(@v1.12) pkg> add MLDatasets
(@v1.12) pkg> add CUDA
(@v1.12) pkg> add Statistics
(@v1.12) pkg> add Random
(@v1.12) pkg> add ImageTransformations
(@v1.12) pkg> add LinearAlgebra
(@v1.12) pkg> add Plots

(@v1.12) pkg> add VideoIO
(@v1.12) pkg> add ImageView
(@v1.12) pkg> add Dates
(@v1.12) pkg> add Gtk
(@v1.12) pkg> add PlotlyJS
(@v1.12) pkg> add cuDNN
(@v1.12) pkg> add GLib
(@v1.12) pkg> add GNNlib
(@v1.12) pkg> add PGLib
(@v1.12) pkg> add GenLib
(@v1.12) pkg> add GtkDrawingArea
(@v1.12) pkg> add ColorSchemes
(@v1.12) pkg> add Optimisers
(@v1.12) pkg> add Colors
(@v1.12) pkg> add Glob
(@v1.12) pkg> add TOML

(@v1.12) pkg> add HTTP
(@v1.12) pkg> add JSON3
 

  


```

 GLib
 PGLib GenLib MHLib MQLib TPLib Git GLNS GLTF GLM GLPK Libz GLFW Glob GZip DTALib LRSLib CDDLib GNNlib

## **Executar o arquivo**

```
julia> ]
pkg> activate ./calculos
julia > import calculos;
julia > calculos.calcular();
julia > calculos.greet();
```
