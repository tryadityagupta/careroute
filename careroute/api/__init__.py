"""HTTP layer.

    app.py       create_app(): routes, CORS, health and version
    schemas.py   request bodies
    services.py  CareService (/care) and ChatService (/chat): one turn, end to end
    location.py  LocationResolver: typed place / GPS -> coordinates
    errors.py    dependency outages -> 503 with the emergency number
"""
