-- Favored Index Type: GiST
-- Reason: Efficient for fuzzy string matching

select *
from (SELECT product_id, name, description
      FROM Products
      WHERE 'heavy' % name) pind;

-- Required Index:
-- CREATE INDEX gist_products_name_trigram ON Products USING GIST (name gist_trgm_ops);