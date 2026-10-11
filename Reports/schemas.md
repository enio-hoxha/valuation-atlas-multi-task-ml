# Database Schemas

## Table: appearances
| Column | Type | Null | Key | Default | Extra |
| --- | --- | --- | --- | --- | --- |
| appearance_id | VARCHAR | YES | None | None | None |
| game_id | INTEGER | YES | None | None | None |
| player_id | INTEGER | YES | None | None | None |
| player_club_id | INTEGER | YES | None | None | None |
| player_current_club_id | VARCHAR | YES | None | None | None |
| date | DATE | YES | None | None | None |
| player_name | VARCHAR | YES | None | None | None |
| competition_id | VARCHAR | YES | None | None | None |
| yellow_cards | INTEGER | YES | None | None | None |
| red_cards | INTEGER | YES | None | None | None |
| goals | INTEGER | YES | None | None | None |
| assists | INTEGER | YES | None | None | None |
| minutes_played | INTEGER | YES | None | None | None |

## Table: club_games
| Column | Type | Null | Key | Default | Extra |
| --- | --- | --- | --- | --- | --- |
| game_id | VARCHAR | YES | None | None | None |
| club_id | INTEGER | YES | None | None | None |
| own_goals | INTEGER | YES | None | None | None |
| own_position | INTEGER | YES | None | None | None |
| own_manager_name | VARCHAR | YES | None | None | None |
| opponent_id | INTEGER | YES | None | None | None |
| opponent_goals | INTEGER | YES | None | None | None |
| opponent_position | INTEGER | YES | None | None | None |
| opponent_manager_name | VARCHAR | YES | None | None | None |
| hosting | VARCHAR | YES | None | None | None |
| is_win | INTEGER | YES | None | None | None |

## Table: clubs
| Column | Type | Null | Key | Default | Extra |
| --- | --- | --- | --- | --- | --- |
| club_id | VARCHAR | YES | None | None | None |
| club_code | VARCHAR | YES | None | None | None |
| name | VARCHAR | YES | None | None | None |
| domestic_competition_id | VARCHAR | YES | None | None | None |
| total_market_value | FLOAT | YES | None | None | None |
| squad_size | INTEGER | YES | None | None | None |
| average_age | FLOAT | YES | None | None | None |
| foreigners_number | INTEGER | YES | None | None | None |
| foreigners_percentage | FLOAT | YES | None | None | None |
| national_team_players | INTEGER | YES | None | None | None |
| stadium_name | VARCHAR | YES | None | None | None |
| stadium_seats | INTEGER | YES | None | None | None |
| net_transfer_record | VARCHAR | YES | None | None | None |
| coach_name | VARCHAR | YES | None | None | None |
| last_season | VARCHAR | YES | None | None | None |
| filename | VARCHAR | YES | None | None | None |
| url | VARCHAR | YES | None | None | None |

## Table: competitions
| Column | Type | Null | Key | Default | Extra |
| --- | --- | --- | --- | --- | --- |
| competition_id | VARCHAR | YES | None | None | None |
| competition_code | VARCHAR | YES | None | None | None |
| name | VARCHAR | YES | None | None | None |
| sub_type | VARCHAR | YES | None | None | None |
| type | VARCHAR | YES | None | None | None |
| country_id | INTEGER | YES | None | None | None |
| country_name | VARCHAR | YES | None | None | None |
| domestic_league_code | VARCHAR | YES | None | None | None |
| confederation | VARCHAR | YES | None | None | None |
| total_clubs | INTEGER | YES | None | None | None |
| url | VARCHAR | YES | None | None | None |

## Table: countries
| Column | Type | Null | Key | Default | Extra |
| --- | --- | --- | --- | --- | --- |
| country_id | VARCHAR | YES | None | None | None |
| country_name | VARCHAR | YES | None | None | None |
| country_code | VARCHAR | YES | None | None | None |
| confederation | VARCHAR | YES | None | None | None |
| total_clubs | INTEGER | YES | None | None | None |
| total_players | INTEGER | YES | None | None | None |
| average_age | FLOAT | YES | None | None | None |
| url | VARCHAR | YES | None | None | None |

## Table: game_events
| Column | Type | Null | Key | Default | Extra |
| --- | --- | --- | --- | --- | --- |
| game_event_id | VARCHAR | YES | None | None | None |
| date | DATE | YES | None | None | None |
| game_id | VARCHAR | YES | None | None | None |
| minute | INTEGER | YES | None | None | None |
| type | VARCHAR | YES | None | None | None |
| club_id | INTEGER | YES | None | None | None |
| club_name | VARCHAR | YES | None | None | None |
| player_id | INTEGER | YES | None | None | None |
| description | VARCHAR | YES | None | None | None |
| player_in_id | INTEGER | YES | None | None | None |
| player_assist_id | INTEGER | YES | None | None | None |

## Table: game_lineups
| Column | Type | Null | Key | Default | Extra |
| --- | --- | --- | --- | --- | --- |
| game_lineups_id | VARCHAR | YES | None | None | None |
| date | DATE | YES | None | None | None |
| game_id | INTEGER | YES | None | None | None |
| player_id | INTEGER | YES | None | None | None |
| club_id | INTEGER | YES | None | None | None |
| player_name | VARCHAR | YES | None | None | None |
| type | VARCHAR | YES | None | None | None |
| position | VARCHAR | YES | None | None | None |
| number | VARCHAR | YES | None | None | None |
| team_captain | INTEGER | YES | None | None | None |

## Table: games
| Column | Type | Null | Key | Default | Extra |
| --- | --- | --- | --- | --- | --- |
| game_id | VARCHAR | YES | None | None | None |
| competition_id | VARCHAR | YES | None | None | None |
| season | VARCHAR | YES | None | None | None |
| round | VARCHAR | YES | None | None | None |
| date | DATE | YES | None | None | None |
| home_club_id | INTEGER | YES | None | None | None |
| away_club_id | INTEGER | YES | None | None | None |
| home_club_goals | INTEGER | YES | None | None | None |
| away_club_goals | INTEGER | YES | None | None | None |
| home_club_position | INTEGER | YES | None | None | None |
| away_club_position | INTEGER | YES | None | None | None |
| home_club_manager_name | VARCHAR | YES | None | None | None |
| away_club_manager_name | VARCHAR | YES | None | None | None |
| stadium | VARCHAR | YES | None | None | None |
| attendance | INTEGER | YES | None | None | None |
| referee | VARCHAR | YES | None | None | None |
| url | VARCHAR | YES | None | None | None |
| home_club_formation | VARCHAR | YES | None | None | None |
| away_club_formation | VARCHAR | YES | None | None | None |
| home_club_name | VARCHAR | YES | None | None | None |
| away_club_name | VARCHAR | YES | None | None | None |
| aggregate | VARCHAR | YES | None | None | None |
| competition_type | VARCHAR | YES | None | None | None |

## Table: national_teams
| Column | Type | Null | Key | Default | Extra |
| --- | --- | --- | --- | --- | --- |
| national_team_id | VARCHAR | YES | None | None | None |
| name | VARCHAR | YES | None | None | None |
| team_code | VARCHAR | YES | None | None | None |
| country_id | VARCHAR | YES | None | None | None |
| country_name | VARCHAR | YES | None | None | None |
| country_code | VARCHAR | YES | None | None | None |
| confederation | VARCHAR | YES | None | None | None |
| team_image_url | VARCHAR | YES | None | None | None |
| squad_size | INTEGER | YES | None | None | None |
| average_age | FLOAT | YES | None | None | None |
| foreigners_number | INTEGER | YES | None | None | None |
| foreigners_percentage | FLOAT | YES | None | None | None |
| total_market_value | INTEGER | YES | None | None | None |
| coach_name | VARCHAR | YES | None | None | None |
| fifa_ranking | INTEGER | YES | None | None | None |
| last_season | VARCHAR | YES | None | None | None |
| url | VARCHAR | YES | None | None | None |

## Table: player_valuations
| Column | Type | Null | Key | Default | Extra |
| --- | --- | --- | --- | --- | --- |
| player_id | INTEGER | YES | None | None | None |
| date | DATE | YES | None | None | None |
| market_value_in_eur | INTEGER | YES | None | None | None |
| current_club_name | VARCHAR | YES | None | None | None |
| current_club_id | INTEGER | YES | None | None | None |
| player_club_domestic_competition_id | VARCHAR | YES | None | None | None |

## Table: players
| Column | Type | Null | Key | Default | Extra |
| --- | --- | --- | --- | --- | --- |
| player_id | INTEGER | YES | None | None | None |
| first_name | VARCHAR | YES | None | None | None |
| last_name | VARCHAR | YES | None | None | None |
| name | VARCHAR | YES | None | None | None |
| last_season | VARCHAR | YES | None | None | None |
| current_club_id | VARCHAR | YES | None | None | None |
| player_code | VARCHAR | YES | None | None | None |
| country_of_birth | VARCHAR | YES | None | None | None |
| city_of_birth | VARCHAR | YES | None | None | None |
| country_of_citizenship | VARCHAR | YES | None | None | None |
| date_of_birth | TIMESTAMP | YES | None | None | None |
| sub_position | VARCHAR | YES | None | None | None |
| position | VARCHAR | YES | None | None | None |
| foot | VARCHAR | YES | None | None | None |
| height_in_cm | INTEGER | YES | None | None | None |
| contract_expiration_date | TIMESTAMP | YES | None | None | None |
| agent_name | VARCHAR | YES | None | None | None |
| image_url | VARCHAR | YES | None | None | None |
| international_caps | INTEGER | YES | None | None | None |
| international_goals | INTEGER | YES | None | None | None |
| current_national_team_id | VARCHAR | YES | None | None | None |
| url | VARCHAR | YES | None | None | None |
| current_club_domestic_competition_id | VARCHAR | YES | None | None | None |
| current_club_name | VARCHAR | YES | None | None | None |
| market_value_in_eur | INTEGER | YES | None | None | None |
| highest_market_value_in_eur | INTEGER | YES | None | None | None |

## Table: transfers
| Column | Type | Null | Key | Default | Extra |
| --- | --- | --- | --- | --- | --- |
| player_id | INTEGER | YES | None | None | None |
| transfer_date | DATE | YES | None | None | None |
| transfer_season | VARCHAR | YES | None | None | None |
| from_club_id | INTEGER | YES | None | None | None |
| to_club_id | INTEGER | YES | None | None | None |
| from_club_name | VARCHAR | YES | None | None | None |
| to_club_name | VARCHAR | YES | None | None | None |
| transfer_fee | DECIMAL(18,3) | YES | None | None | None |
| market_value_in_eur | DECIMAL(18,3) | YES | None | None | None |
| player_name | VARCHAR | YES | None | None | None |

## Table: version
| Column | Type | Null | Key | Default | Extra |
| --- | --- | --- | --- | --- | --- |
| commit_hash | VARCHAR | YES | None | None | None |

