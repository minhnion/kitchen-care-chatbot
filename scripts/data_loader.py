from pymongo import MongoClient
from app.config import settings

def get_db_connection():
    try:
        client = MongoClient(settings.MONGODB_URI)
        client.admin.command('ping')
        return client
    except Exception as e:
        return None
    
def fetch_populated_products(client: MongoClient):
    db = client[settings.DB_NAME]
    products_collection = db[settings.PRODUCT_COLLECTION]
    
    pipeline = [
        #Phase 1: Filter products with is not Delete
        {
            '$match': { 'isDelete': False }
        },
        
        #Phase 2: Join with parent collection
        
        {
            '$lookup': {
                'from': 'categories',
                'localField': 'category',
                'foreignField': '_id',
                'as': 'categoryDetails'
            }
        },
        
        {
            '$lookup': {
                'from': 'subcategories',
                'localField': 'subCategory',
                'foreignField': '_id',
                'as': 'subCategoryDetails'
            }
        },
        
        {
            '$lookup': {
                'from': 'brands',
                'localField': 'brand',
                'foreignField': '_id',
                'as': 'brandDetails'
            }
        },
        
        #Phase 3: Re format output
        {
            '$project': {
                '_id': 1, 'name': 1, 'description': 1, 'price': 1, 
                'specifications': 1, 'slug': 1, 'mainImage': 1,
                
                'categoryName': { '$arrayElemAt': ['$categoryDetails.name', 0]},
                'subCategoryName': { '$arrayElemAt': ['$subCategoryDetails.name', 0] },
                'brandName': { '$arrayElemAt': ['$brandDetails.name', 0] },
            }
        }
    ]
    
    populated_products = list(products_collection.aggregate(pipeline))
    print(f"Found and populated {len(populated_products)} products")
    
    return populated_products