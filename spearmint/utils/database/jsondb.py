
import sys
import time
import os
from abstractdb                  import AbstractDB
from spearmint.utils.compression import compress_nested_container, decompress_nested_container
import json
from filelock import FileLock

class JsonDB(AbstractDB):

    def __init__(self, database_address='anythingbutmongo', database_name='spearmint'):
    
        self.client = None
        self.db_adress    = os.getcwd() + '/' + database_address +  '.json'
        self.lock_adress = self.db_adress + '.lock'
        self.lock = FileLock(self.lock_adress)
        with self.lock:
            try:
                with open(self.db_adress, 'r') as f:
                    self.db = json.load(f)
            except:
                self.db = {}
                self.db['Placeholder'] = 0
                with open(self.db_adress, 'w') as f:
                    json.dump(self.db, f)
                
    def save(self, save_doc, experiment_name, experiment_field, field_filters=None):
        """
        Saves a document into the database.
        Compresses any numpy arrays so that they can be saved to MongoDB.
        field_filters must return at most one document, otherwise it is not clear
        which one to update and an exception will be raised.
        """
        with self.lock:
            with open(self.db_adress, 'r') as f:
                self.db = json.load(f)
            
            if not self.db.get(experiment_name, False):
                self.db[experiment_name] = {}
            
            if not self.db[experiment_name].get(experiment_field, False):
                self.db[experiment_name][experiment_field] = []
            
            dbcollection = self.db[experiment_name][experiment_field]
            # the document to be saved
            save_doc = compress_nested_container(save_doc)

            # if they give field filters {'id': 1}
            if field_filters is not None:
                field_key = list(field_filters.keys())[0]
                field_value = list(field_filters.values())[0]
                
                # check if it's there and should be replaced
                for i in range(len(dbcollection)):

                    if dbcollection[i][field_key] == field_value:
                        dbcollection[i] = save_doc
                        with open(self.db_adress, 'w') as f:
                            json.dump(self.db, f)
                        return
                # otherwise it should be appended (either it's empty or should be appended anyway)
                dbcollection.append(save_doc)
                with open(self.db_adress, 'w') as f:
                    json.dump(self.db, f)
            
            # or if there was nothing specified at all, overwrite
            else:
                self.db[experiment_name][experiment_field] = [save_doc]
                with open(self.db_adress, 'w') as f:
                    json.dump(self.db, f)
        
    def load(self, experiment_name, experiment_field, field_filters=None):
        # Return a list of documents from the database, decompressing any numpy arrays
        with self.lock: 
            with open(self.db_adress, 'r') as f:
                self.db = json.load(f)
            
        # if no experiment name, initiate
        if not self.db.get(experiment_name, False):
            self.db[experiment_name] = {}

        # if hypers doesn't exist, return None
        if not self.db[experiment_name].get(experiment_field, False):
            return None

        dbcollection = self.db[experiment_name][experiment_field]
        ### This should only make a difference when {} is entered! that's the issue
        if field_filters is None:
            
            if len(dbcollection) == 0:
                return None

            if len(dbcollection) == 1:
                return decompress_nested_container(dbcollection[0])
            
            else:
                return [decompress_nested_container(val) for val in dbcollection]
        else:
            field_key = list(field_filters.keys())[0]
            field_value = list(field_filters.values())[0]

            for i in range(len(dbcollection)):
                
                if dbcollection[i][field_key] == field_value:
                    return decompress_nested_container(dbcollection[i])


    def remove(self, experiment_name, experiment_field, field_filters={}):
        print('Trying to remove')
        self.db[experiment_name][experiment_field].remove(field_filters)


    def get_database(self):
        return self.db
