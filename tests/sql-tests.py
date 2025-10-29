import mysql.connector

mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="parking"
)

mycursor = mydb.cursor()


query = "INSERT INTO lot1 (is_occupied) VALUES (1)"

mycursor.execute(query)

mydb.commit()


mycursor.execute("SELECT * from lot1")
for x in mycursor:
    print(x)
