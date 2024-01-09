from ADAM.import_mods import install_mod
from ADAM.functions import list_available_commands, toggle_lockdown, lock_bot, unlock_bot
from ADAM.Hippocampus import update_user_profile, create_new_bot
from ADAM.respond import copy_voice
from ADAM.idle import idle_settings
from ADAM.mod_builder import mod_builder

COMMANDS = [
    {"commands": ["install mod"], "action": install_mod},
    {"commands": ["what can you do?"], "action": list_available_commands, "args": (user_profile,)},
    {"commands": ["update my profile"], "action": update_user_profile, "args": (user_profile,)},
    {"commands": ["build a bot"], "action": create_new_bot, "args": (user_id, user_profile)},
    {"commands": ["copy this voice"], "action": copy_voice, "args": (bot_profile,)},
    {"commands": ["lockdown"], "action": toggle_lockdown, "args": ()},
    {"commands": ["lock yourself"], "action": lock_bot, "args": (bot_profile,)},
    {"commands": ["unlock yourself"], "action": unlock_bot, "args": (bot_profile,)},
    {"commands": ["idle chat settings"], "action": idle_settings, "args": (bot_profile,)},
    {"commands": ["the mod builder"], "action": mod_builder, "args": ()},
]
MAP = {
    install_mod: (bot_profile, user_profile),
    list_available_commands: (user_profile,),
    update_user_profile: (user_profile,),
    create_new_bot: (user_id, user_profile),
    copy_voice: (bot_profile,),
    toggle_lockdown: (),
    lock_bot: (bot_profile,),
    unlock_bot: (bot_profile,),
    mod_builder: (),
    idle_settings: (bot_profile,),
}