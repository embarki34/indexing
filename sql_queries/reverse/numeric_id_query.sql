-- Favored Index Type: Reverse
-- Reason: Good for high-concurrency environments with sequential IDs

SELECT user_id, first_name, last_name, email
FROM Users
WHERE user_id = 1233;

-- Required Index:
-- CREATE INDEX reverse_users_id ON Users USING BTREE (user_id) REVERSE; 