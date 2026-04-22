## 📁 هيكل المشروع

```
student_registration/
├── app.py               ← Flask application (routes + logic)
├── database.py          ← DB init, schema, helper
├── wsgi.py              ← Production WSGI entry point
├── Procfile             ← Railway / Heroku deployment
├── requirements.txt     ← Python dependencies
├── .env                 ← Secret config (DO NOT commit)
├── instance/
│   └── students.db      ← SQLite database (auto-created)
├── static/
│   └── uploads/         ← Student photos (JPG)
└── templates/
    ├── index.html        ← Public registration form
    ├── admin_login.html  ← Admin login page
    └── admin_dashboard.html ← Admin panel
```
