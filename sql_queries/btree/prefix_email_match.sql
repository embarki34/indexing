-- Favored Index Type: B-Tree
-- Reason: Efficient for prefix matching and LIKE operations

SELECT user_id, first_name, last_name, email
FROM Users
WHERE email LIKE 'john%';

-- Required Index:
-- CREATE INDEX btree_users_email ON Users USING BTREE (email); 