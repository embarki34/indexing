from google import genai
import pandas as pd
import time
import re
import os
from dotenv import load_dotenv
import requests

client = genai.Client(api_key="AIzaSyAXLjLGb0bUbjfgvFKE-6E66sqMhIfPKqY")
sql_content = """




-- Dumping database structure for educ_market
CREATE DATABASE IF NOT EXISTS `educ_market` /*!40100 DEFAULT CHARACTER SET utf8mb4 */;
USE `educ_market`;

-- Dumping structure for table educ_market.categories
CREATE TABLE IF NOT EXISTS `categories` (
  `category_id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(100) NOT NULL,
  `description` text DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`category_id`)
) ENGINE=InnoDB AUTO_INCREMENT=21 DEFAULT CHARSET=utf8mb4;

-- Data exporting was unselected.

-- Dumping structure for table educ_market.orderitems
CREATE TABLE IF NOT EXISTS `orderitems` (
  `order_item_id` int(11) NOT NULL AUTO_INCREMENT,
  `order_id` int(11) DEFAULT NULL,
  `product_id` int(11) DEFAULT NULL,
  `quantity` int(11) DEFAULT 1,
  `price` decimal(10,2) NOT NULL,
  PRIMARY KEY (`order_item_id`),
  KEY `order_id` (`order_id`),
  KEY `product_id` (`product_id`),
  CONSTRAINT `orderitems_ibfk_1` FOREIGN KEY (`order_id`) REFERENCES `orders` (`order_id`),
  CONSTRAINT `orderitems_ibfk_2` FOREIGN KEY (`product_id`) REFERENCES `products` (`product_id`)
) ENGINE=InnoDB AUTO_INCREMENT=1425909 DEFAULT CHARSET=utf8mb4;

-- Data exporting was unselected.

-- Dumping structure for table educ_market.orders
CREATE TABLE IF NOT EXISTS `orders` (
  `order_id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) DEFAULT NULL,
  `order_date` timestamp NOT NULL DEFAULT current_timestamp(),
  `status` enum('Pending','Shipped','Delivered','Cancelled') DEFAULT 'Pending',
  `total_amount` decimal(10,2) NOT NULL,
  `shipping_address` text DEFAULT NULL,
  PRIMARY KEY (`order_id`),
  KEY `user_id` (`user_id`),
  CONSTRAINT `orders_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`)
) ENGINE=InnoDB AUTO_INCREMENT=474783 DEFAULT CHARSET=utf8mb4;

-- Data exporting was unselected.

-- Dumping structure for table educ_market.products
CREATE TABLE IF NOT EXISTS `products` (
  `product_id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `description` text DEFAULT NULL,
  `price` decimal(10,2) NOT NULL,
  `stock_quantity` int(11) DEFAULT 0,
  `category_id` int(11) DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`product_id`),
  KEY `category_id` (`category_id`),
  CONSTRAINT `products_ibfk_1` FOREIGN KEY (`category_id`) REFERENCES `categories` (`category_id`)
) ENGINE=InnoDB AUTO_INCREMENT=2001 DEFAULT CHARSET=utf8mb4;

-- Data exporting was unselected.

-- Dumping structure for table educ_market.users
CREATE TABLE IF NOT EXISTS `users` (
  `user_id` int(11) NOT NULL AUTO_INCREMENT,
  `first_name` varchar(100) DEFAULT NULL,
  `last_name` varchar(100) DEFAULT NULL,
  `email` varchar(255) NOT NULL,
  `password` varchar(255) NOT NULL,
  `phone_number` varchar(20) DEFAULT NULL,
  `shipping_address` text DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`user_id`)
) ENGINE=InnoDB AUTO_INCREMENT=10001 DEFAULT CHARSET=utf8mb4;




"""
# print(sql_content)
index_types = "btree,gist,hash,gin,bitmap,reverse"

prompt = f"""
        Based on the following SQL schema:

        ```sql
        {sql_content}
        ```

        Generate a concise SQL SELECT query to interrogate the data (e.g., find specific records, aggregate data, etc.).
        Specify the preferred method of indexing for the query. 
        Do NOT create tables or indexes. Focus on SELECT queries for data interrogation.
        This is query number i for index type '{index_types}'. Make each query different.
        Return the response in the format: "quiri: string; preferindex: string".
        """

while True:
    for index_type in index_types.split(','):
        
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=AIzaSyAXLjLGb0bUbjfgvFKE-6E66sqMhIfPKqY",
            headers={'Content-Type': 'application/json'},
            json={
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
        )
        query_text = response.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
        with open("query.txt", "r") as f:
            existing_queries = f.read().splitlines()
        
        if query_text and query_text not in existing_queries:
            with open("query.txt", "a") as f:
                f.write(query_text + "\n")
    
    time.sleep(30)  # Wait for 30 seconds before the next iteration
