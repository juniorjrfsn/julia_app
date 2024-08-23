using CSV, SQLite, DataFrames

function query_data(question, db_name::String; table_name="training_data")
    """
    Queries the SQLite database for data matching the question.

    Args:
        question: The question to query.
        db_name: The name of the SQLite database.
        table_name: The name of the table in the database.

    Returns:
        A DataFrame containing the results of the query.
    """

    DB = SQLite.DB(db_name)
    query_result = DB.query("SELECT * FROM $table_name WHERE ...") # Replace ... with your query logic
    data = DataFrame(query_result)
    close(DB)

    return data
end

# Example usage:
result = query_data("SELECT * FROM training_data WHERE x > 2", "my_database.db")
println(result)