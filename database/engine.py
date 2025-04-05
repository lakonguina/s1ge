import os
from dotenv import load_dotenv

from sqlmodel import create_engine, Session, SQLModel, select

load_dotenv()

engine = create_engine(os.getenv("POSTGRES_URL"))

def get_session():
    with Session(engine) as session:
        yield session

def get_or_create(session: Session, model: SQLModel, query_keys: list, **kwargs) -> SQLModel:
    """
    Retrieve an existing instance of the model or create a new one.

    Parameters:
    - session: The SQLModel session to use.
    - model: The SQLModel class to query.
    - query_keys: A list of keys from kwargs to use as the query condition.
                   Example: ['id'] to use kwargs['id'] for the query.
    - **kwargs: Additional keyword arguments to create a new instance if not found.

    Returns:
    - SQLModel: The found or newly created instance.
    """
    # Build query conditions properly using SQLModel's syntax
    query = select(model)
    for key in query_keys:
        if key in kwargs:
            # Get the column attribute from the model class
            column = getattr(model, key)
            # Add the filter condition
            query = query.where(column == kwargs[key])
    # Execute the query
    instance = session.exec(query).first()
    if instance:
        return instance
    else:
        instance = model(**kwargs)
        session.add(instance)
        session.flush()
        return instance