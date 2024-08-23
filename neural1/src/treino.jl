using CSV, SQLite

function store_training_data(data, filename::String; db_name::String="")
    """
    Stores training data in either a CSV file or an SQLite database.

    Args:
        data: A DataFrame containing the training data.
        filename: The name of the CSV file to save.
        db_name: The name of the SQLite database to create.
    """

    if db_name == ""
        # Save to CSV
        CSV.write(filename, data)
        println("Data saved to CSV file: ", filename)
    else
        # Save to SQLite
        DB = SQLite.DB(db_name)
        SQLite.load!(DB, data, "training_data")
        println("Data saved to SQLite database: ", db_name)
        close(DB)
    end
end

# Example usage:
data = DataFrame(x=[1,2,3], y=[4,5,6])
store_training_data(data, "training_data.csv", "my_database.db")