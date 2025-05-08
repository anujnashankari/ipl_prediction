from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    TeamViewSet, PlayerViewSet, VenueViewSet, MatchViewSet,
    PlayerPerformanceViewSet, PredictionViewSet, PlayerPredictionViewSet
)

router = DefaultRouter()
router.register(r'teams', TeamViewSet)
router.register(r'players', PlayerViewSet)
router.register(r'venues', VenueViewSet)
router.register(r'matches', MatchViewSet)
router.register(r'performances', PlayerPerformanceViewSet)
router.register(r'predictions', PredictionViewSet)
router.register(r'player-predictions', PlayerPredictionViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
