"""
Include file for interfacing with the parking lot database
"""

import mariadb

conn_params= {
    "user" : "Curtis_Laptop",
    "password" : "root",
    "host" : "172.31.63.121",
    "port" : 3306,
    "database" : "parking"
}

def markSpotInDB(spotID, occupied):
    spotID = spotID + 1 # count in the DB starts from 1

    with mariadb.connect(**conn_params) as conn:
        with conn.cursor() as cursor:

            data = (spotID,)

            if occupied:
                sql = "UPDATE lot1 SET is_occupied = 1 WHERE spot_id = ?"
            else:
                sql = "UPDATE lot1 SET is_occupied = 0 WHERE spot_id = ?"
            
            cursor.execute(sql, data)

            conn.commit()

def processList(spaces):
    
    for s in spaces:
        if s.occupied:
            print("space " ,s.id, "is OCCUPIED")
            markSpotInDB(s.id, True)
        else:
            print("space " ,s.id, "is VACANT")
            markSpotInDB(s.id, False)
