import mysql.connector


def mydb():
    conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="furesign"
    )
    return conn


