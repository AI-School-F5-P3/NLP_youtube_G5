import time
import MySQLdb
import os

def wait_for_db():
    while True:
        try:
            MySQLdb.connect(
                host=os.getenv('DB_HOST'),
                user=os.getenv('DB_USER'),
                passwd=os.getenv('DB_PASSWORD'),
                db=os.getenv('DB_NAME'),
                port=int(os.getenv('DB_PORT', 3306))
            )
            break
        except MySQLdb.Error:
            print("Database not ready yet. Waiting...")
            time.sleep(1)

if __name__ == "__main__":
    wait_for_db()
    print("Database is ready!")