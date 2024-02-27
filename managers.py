import pymongo
import bcrypt
import ast
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class LoginManager:

    def __init__(self) -> None:
        # MongoDB connection
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["project"]
        self.collection = self.db["users"]
        self.salt = b"$2b$12$ezgTynDsK3pzF8SStLuAPO"  # TODO: if not working, generate a new salt

    def register_user(self, username: str, password: str) -> None:
        # Check if username and password are not empty strings
        if not username or not password:
            raise ValueError("Username and password are required")
        
        # Check if the length of both username and password is at least 3 characters
        if len(username) < 3 or len(password) < 3:
            raise ValueError("Username and password must be at least 3 characters")
        
        # Check if the username already exists in the database
        user = self.collection.find_one({"username": username})
        if user:
            raise ValueError(f"User already exists: {username}")
        
        # Hash the provided password using bcrypt
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), self.salt).decode('utf-8')
        
        # Create a new user in the database
        self.collection.insert_one({"username": username, "password": hashed_password, "rented_games": [], "favorite_genre": "", "favorite_game": ""})
    

    def login_user(self, username: str, password: str) -> object:    
        # Hash the provided password using bcrypt
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), self.salt).decode('utf-8')
        
        # Query the MongoDB collection to find a user with the provided username and hashed password
        user = self.collection.find_one({"username": username, "password": hashed_password})
        
        # If a user is found, print "Logged in successfully as: {username}" and return the user object
        if user:
            print(f"Logged in successfully as: {username}")
            return user
        else:
            raise ValueError("Invalid username or password")


class DBManager:

    def __init__(self) -> None:
        # MongoDB connection
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["project"]
        self.user_collection = self.db["users"]
        self.game_collection = self.db["games"]

    def load_csv(self) -> None:
        # Loads data from a CSV file into games collection.
        # Parameters:
        # • None
        # Returns:
        # • None
        # Instructions:
        # 1. Load the CSV file "NintendoGames.csv".
        # 2. Insure "genres" field will store as list (u can use ast.literal_eval for this).
        # 3. Add "is_rented" field with the values False.
        # 4. Insert all the records into games collection.
        # 5. Insure that using this function few times in a row will not insert duplicate items to
        # the collection.
        
        # if self.game_collection.count > 0:
        #     return
        
        if self.game_collection.count_documents({}) > 0:
            return
        
        # Load the CSV file "NintendoGames.csv"
        df = pd.read_csv("NintendoGames.csv")
        
        # Insure "genres" field will store as list (u can use ast.literal_eval for this)
        df["genres"] = df["genres"].apply(ast.literal_eval)
        
        # Add "is_rented" field with the values False
        df["is_rented"] = False
        
        # Insert all the records into games collection
        self.game_collection.insert_many(df.to_dict("records"))
        
        # Insure that using this function few times in a row will not insert duplicate items to the collection
        self.game_collection.create_index("title", unique=True)
        

    def recommend_games_by_genre(self, user: dict) -> str:
        # Recommends games based on the user's rented game genre. Don’t recommend games that
        # are already owned.
        # Parameters:
        # • user (dict): The user object.
        # Returns:
        # • str: A string containing recommended game titles based on genre.
        # Instructions:
        # 1. Get the list of games rented by the user from the user object.
        # 2. If no games are rented, return "No games rented".
        # 3. Select a genre randomly from the pool of rented games, taking into account the
        # probability distribution. For instance, if there are three games categorized as
        # "shooter" and one as "RPG," the likelihood of selecting "shooter" would be 75%.
        # 4. Query the game collection to find 5 random games with the chosen genre.
        # 5. Return the titles as a string separated with "\n".
        
        # Get the list of games rented by the user from the user object
        rented_games = user["rented_games"]
        
        # If no games are rented, return "No games rented"
        if not rented_games:
            return "No games rented"
        
        # Select a genre randomly from the pool of rented games, taking into account the probability distribution
        genres = self.game_collection.aggregate([
            {"$match": {"title": {"$in": rented_games}}},
            {"$unwind": "$genres"},
            {"$group": {"_id": "$genres", "count": {"$sum": 1}}},
            {"$project": {"_id": 0, "genre": "$_id", "count": 1}}
        ])
        
        genres = pd.DataFrame(genres)
        genres["probability"] = genres["count"] / genres["count"].sum()
        chosen_genre = genres.sample(weights="probability").iloc[0]["genre"]
        
        # Query the game collection to find 5 random games with the chosen genre
        games = self.game_collection.find({"genres": chosen_genre, "title": {"$nin": rented_games}}).limit(5)
        
        # Return the titles as a string separated with "\n"
        return "\n".join([game["title"] for game in games])
        

    def recommend_games_by_name(self, user: dict) -> str:
        # Recommends games based on random user's rented game name. Don’t recommend games
        # that are already owned.
        # Parameters:
        # • user (dict): The user object.
        # Returns:
        # • str: A string containing recommended game titles based on similarity.
        # Instructions:
        # 1. Get the list of games rented by the user from the user object.
        # 2. If no games are rented, return "No games rented".
        # 3. Choose a random game from the rented games.
        # 4. Compute TF-IDF vectors for all game titles and the chosen title (u can use
        # TfidfVectorizer from sklearn library).
        # 5. Compute cosine similarity between the TF-IDF vectors of the chosen title and all
        # other games (u can use cosine_similarity from sklearn library).
        # 6. Sort the titles based on cosine similarity and return the top 5 recommended titles as
        # a string separated with "\n".
        
        # Get the list of games rented by the user from the user object
        rented_games = user["rented_games"]
        
        # If no games are rented, return "No games rented"
        if not rented_games:
            return "No games rented"
        
        # Choose a random game from the rented games
        chosen_game = random.choice(rented_games)
        
        # Compute TF-IDF vectors for all game titles and the chosen title (u can use TfidfVectorizer from sklearn library)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.game_collection.distinct("title"))
        chosen_game_vector = vectorizer.transform([chosen_game])
        
        # Compute cosine similarity between the TF-IDF vectors of the chosen title and all other games (u can use cosine_similarity from sklearn library)
        cosine_sim = cosine_similarity(chosen_game_vector, tfidf_matrix)
        
        # Sort the titles based on cosine similarity and return the top 5 recommended titles as a string separated with "\n"
        recommended_games = cosine_sim.argsort()[0][-6:-1]
        return "\n".join(self.game_collection.distinct("title")[recommended_games])

    def rent_game(self, user: dict, game_title: str) -> str:
        # Rents a game for the user.
        # Parameters:
        # • user (dict): The user object.
        # • game_title (str): The title of the game to be rented.
        # Returns:
        # • str: A message indicating the success or failure of the rental process.
        # Instructions:
        # 1. Query the game collection to find the game with the provided title.
        # 2. If the game is found:
        # • Check if the game is not already rented.
        # • If not rented, mark the game as rented in the game collection and add it to
        # the user's rented games list.
        # • Return "{game_title} rented successfully".
        # 3. If the game is not found, return "{game_title} not found".
        # 4. If the game is already rented, return "{game_title} is already rented".
        
        # Query the game collection to find the game with the provided title
        game = self.game_collection.find_one({"title": game_title})
        
        # If the game is found
        if game:
            # Check if the game is not already rented
            if not game["is_rented"]:
                # Mark the game as rented in the game collection and add it to the user's rented games list
                self.game_collection.update_one({"title": game_title}, {"$set": {"is_rented": True}})
                self.user_collection.update_one({"username": user["username"]}, {"$push": {"rented_games": game_title}})
                return f"{game_title} rented successfully"
            else:
                return f"{game_title} is already rented"
        else:
            return f"{game_title} not found"
        
    def return_game(self, user: dict, game_title: str) -> str:
        # Returns a rented game.
        # Parameters:
        # • user (dict): The user object.
        # • game_title (str): The title of the game to be returned.
        # Returns:
        # • str: A message indicating the success or failure of the return process.
        # Instructions:
        # 1. Get the list of games rented by the user from the user object.
        # 2. If the game with the provided title is rented by the user:
        # • Remove the game from the user's rented games list.
        # • Mark the game as not rented in the game collection.
        # • Return "{game_title} returned successfully".
        # 3. If the game is not rented by the user, return "{game_title} was not rented by you".

        # Get the list of games rented by the user from the user object
        rented_games = user["rented_games"]
        
        # If the game with the provided title is rented by the user
        if game_title in rented_games:
            # Remove the game from the user's rented games list
            self.user_collection.update_one({"username": user["username"]}, {"$pull": {"rented_games": game_title}})
            # Mark the game as not rented in the game collection
            self.game_collection.update_one({"title": game_title}, {"$set": {"is_rented": False}})
            return f"{game_title} returned successfully"
        else:
            return f"{game_title} was not rented by you"