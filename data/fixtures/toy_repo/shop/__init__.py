"""A small layered shop application used as a cross-file retrieval fixture.

Layering mirrors the Controller -> Service -> Repository chain in the README:

    OrderController -> OrderService -> OrderRepository
                                    -> PaymentService -> PaymentRepository
                                    -> UserService    -> UserRepository
"""
