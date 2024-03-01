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
        if not username or not password:
            raise ValueError("Username and password are required")
        
        if len(username) < 3 or len(password) < 3:
            raise ValueError("Username and password must be at least 3 characters")
        
        user = self.collection.find_one({"username": username})
        if user:
            raise ValueError(f"User already exists: {username}")
        
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), self.salt).decode('utf-8')
        
        self.collection.insert_one({"username": username, "password": hashed_password, "rented_games": [], "favorite_genre": "", "favorite_game": ""})
        print(f"User {username} registered successfully")

    def login_user(self, username: str, password: str) -> object:    
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), self.salt).decode('utf-8')
        
        user = self.collection.find_one({"username": username, "password": hashed_password})
        
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
        if self.game_collection.count_documents({}) > 0:
            print("Data already loaded")
            return
        
        df = pd.read_csv("NintendoGames.csv")
        df["genres"] = df["genres"].apply(ast.literal_eval)
        df["is_rented"] = False
        
        for index, row in df.iterrows():
            if self.game_collection.find_one({"title": row["title"]}):
                print(f"Game {row['title']} already exists in the collection")
                continue
            self.game_collection.insert_one(row.to_dict())

        self.game_collection.create_index("title", unique=True)
        print("Data loaded successfully")

    def recommend_games_by_genre(self, user: dict) -> str:
        # Recommends games based on the user's rented game genre. Don’t recommend games that are already owned.
        rented_games = user["rented_games"]
        
        if not rented_games:
            print("No games rented")
            return "No games rented"
        
        genres = self.game_collection.aggregate([
            {"$match": {"title": {"$in": rented_games}}},
            {"$unwind": "$genres"},
            {"$group": {"_id": "$genres", "count": {"$sum": 1}}},
            {"$project": {"_id": 0, "genre": "$_id", "count": 1}}
        ])
        
        genres = pd.DataFrame(genres)
        genres["probability"] = genres["count"] / genres["count"].sum()
        chosen_genre = genres.sample(weights="probability").iloc[0]["genre"]
        
        games = self.game_collection.find({"genres": chosen_genre, "title": {"$nin": rented_games}}).limit(5)
        return_string = "\n".join([game["title"] for game in games])
        print(f"recommend_games_by_genre:\n{return_string}")
        return return_string

    def recommend_games_by_name(self, user: dict) -> str:
        # Recommends games based on random user's rented game name. Don’t recommend games that are already owned.
        rented_games = user["rented_games"]
        
        if not rented_games:
            print("No games rented")
            return "No games rented"
        
        chosen_game = random.choice(rented_games)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.game_collection.distinct("title"))
        chosen_game_vector = vectorizer.transform([chosen_game])
        cosine_sim = cosine_similarity(chosen_game_vector, tfidf_matrix)
        recommended_games = cosine_sim.argsort()[0][-6:-1]
        recommended_games_titles = [self.game_collection.distinct("title")[index] for index in recommended_games]

        return_string = "\n".join(recommended_games_titles)
        print(f"recommend_games_by_name:\n{return_string}")
        return return_string

    def rent_game(self, user: dict, game_title: str) -> str:
        # Rents a game for the user.
        game = self.game_collection.find_one({"title": game_title})
        
        if game:
            if not game["is_rented"]:
                self.game_collection.update_one({"title": game_title}, {"$set": {"is_rented": True}})
                self.user_collection.update_one({"username": user["username"]}, {"$push": {"rented_games": game_title}})
                user["rented_games"].append(game_title)
                return f"{game_title} rented successfully"
            else:
                return f"{game_title} is already rented"
        else:
            return f"{game_title} not found"
        
    def return_game(self, user: dict, game_title: str) -> str:
        # Returns a rented game.
        rented_games = user["rented_games"]

        if game_title in rented_games:
            self.user_collection.update_one({"username": user["username"]}, {"$pull": {"rented_games": game_title}})
            self.game_collection.update_one({"title": game_title}, {"$set": {"is_rented": False}})
            user["rented_games"].remove(game_title)
            return f"{game_title} returned successfully"
        else:
            return f"{game_title} was not rented by you"
