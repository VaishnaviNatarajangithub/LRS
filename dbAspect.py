'''''import pymysql.cursors

class DBConnection():
    def __init__(self):
        self.connection = pymysql.connect(host='localhost', user='root',
            password='')

        with self.connection.cursor() as cursor:
            cursor.execute("CREATE DATABASE IF NOT EXISTS alpr")

        self.connection.commit()

    def save_alpr(self, license_plate_text, moment):
        """
        Saves the license plate text in the database table
        
        Parameters:
        -----------
        license_plate_text: str; the text on the license plate
        moment: str; the current date and time
        """
        try:
            with self.connection.cursor() as cursor:
                table_sql = "CREATE TABLE IF NOT EXISTS `alpr`.`alpr` ( `id` INT NOT NULL AUTO_INCREMENT , `plate_text` VARCHAR(15) NOT NULL , `moment` VARCHAR(30) NOT NULL , PRIMARY KEY (`id`)) ENGINE = InnoDB;"
                cursor.execute(table_sql)
                #insert record
                sql = "INSERT INTO `alpr`.`alpr` (plate_text, moment) VALUES(%s, %s)"
                cursor.execute(sql, (license_plate_text, moment))
            self.connection.commit()

        finally:
            pass'''''

import pandas as pd
import os
import csv
from datetime import datetime

class DBConnection:
    def __init__(self, data_file='vehicles_dataset.xlsx'):
        """
        Initialize DBConnection with local dataset file.
        Supports Excel (.xlsx) or CSV (.csv) based on file extension.
        """
        self.vehicle_df = pd.DataFrame()
        self.data_file = data_file

        if not os.path.exists(data_file):
            print(f"Data file {data_file} not found.")
            return

        try:
            if data_file.lower().endswith('.xlsx') or data_file.lower().endswith('.xls'):
                self.vehicle_df = pd.read_excel(data_file)
            elif data_file.lower().endswith('.csv'):
                self.vehicle_df = pd.read_csv(data_file)
            else:
                print("Unsupported file format. Use Excel (.xlsx) or CSV (.csv).")
        except Exception as e:
            print(f"Error loading data file {data_file}: {e}")
            self.vehicle_df = pd.DataFrame()

    def get_vehicle_info(self, plate_text):
        """
        Search the dataset for a matching registration number.
        Returns a dict with vehicle details or None if not found.
        """
        if self.vehicle_df.empty:
            print("Vehicle dataset is empty or not loaded.")
            return None

        registration_col = 'plate_number'  # Ensure this matches your actual dataset
        owner_col = 'owner'
        issue_col = 'issue_date'
        expiry_col = 'expiry_date'
        chassis_col = 'chasis_number'
        type_col = 'type'

        if registration_col not in self.vehicle_df.columns:
            print(f"Column '{registration_col}' not found in dataset.")
            return None

        matched = self.vehicle_df[self.vehicle_df[registration_col] == plate_text]

        if matched.empty:
            return None

        row = matched.iloc[0]

        return {
            'owner': row.get(owner_col, ''),
            'issue_date': row.get(issue_col, ''),
            'expiry_date': row.get(expiry_col, ''),
            'chassis': row.get(chassis_col, ''),
            'type': row.get(type_col, '')
        }

    def save_alpr(self, plate_text, timestamp=None):
        """
        Save ALPR result to a local CSV log file.
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        filename = 'alpr_log.csv'
        file_exists = os.path.isfile(filename)

        try:
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['Plate Number', 'Timestamp'])
                writer.writerow([plate_text, timestamp])
        except Exception as e:
            print(f"Error saving ALPR result: {e}")
