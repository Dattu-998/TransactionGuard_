from fastapi import Request

from app.predictor import Predictor


def get_predictor(request: Request) -> Predictor:
    return request.app.state.predictor
