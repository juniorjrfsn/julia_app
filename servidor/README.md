# julia_app

## adicionar dependencias
```
pkg> generate servidor

julia> ]

pkg> activate ./servidor
julia > import servidor;
julia > servidor.greet();
julia > ]
pkg> add JSON
pkg> add Pkg
pkg> add SQLite
pkg> add DataFrames


```
## **Executar o arquivo**
```
julia > servidor.greet();
```

# executar
```
	$ cd servidor/
	$ julia main.jl
```