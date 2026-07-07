from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Profile(models.Model):
    TIER_CHOICES = [
        ('free', 'Free'),
        ('pro', 'Pro'),
        ('elite', 'Elite'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    tier = models.CharField(max_length=10, choices=TIER_CHOICES, default='free')

    # Analyzer / Fit Finder usage tracking
    analyzer_finder_queries_used = models.PositiveIntegerField(default=0)
    analyzer_finder_queries_reset_at = models.DateTimeField(null=True, blank=True)

    # Find & Apply usage tracking (paid feature only)
    find_apply_searches_used = models.PositiveIntegerField(default=0)
    find_apply_searches_reset_at = models.DateTimeField(null=True, blank=True)

    # Credit balance for pay-as-you-go purchases
    credits = models.PositiveIntegerField(default=0)

    # Stripe references, filled in once billing is wired up
    stripe_customer_id = models.CharField(max_length=255, blank=True, null=True)
    stripe_subscription_id = models.CharField(max_length=255, blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username} ({self.tier})"
    
from django.db.models.signals import post_save
from django.dispatch import receiver


@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)
    else:
        # Profile might not exist yet for pre-existing users; guard against that
        Profile.objects.get_or_create(user=instance)