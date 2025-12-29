# config_db.py

# ================= MySQL 配置 (t_archives_user) =================
MYSQL_CONFIG = {
    "host": "pc-bp12546upf9pve796.rwlb.rds.aliyuncs.com",
    "port": 3306,
    "user": "dev_health_admin",
    "password": "x1*sl@au^lPoJgE#",
    "database": "sino_ican_user_test",
    "charset": "utf8mb4"
}

# ================= MongoDB 配置 (t_health_cgm_detection_data) =================
MONGO_URI = "mongodb://test-sinocloud-us:5zdQ3wOCFR!$Z7hS@dds-bp103034bf91a434-pub.mongodb.rds.aliyuncs.com:3717/sino-cloud-test?maxIdleTimeMS=3000"

MONGO_DB_NAME = "sino-cloud-test"
MONGO_COLLECTION = "t_health_cgm_detection_data"