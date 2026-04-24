"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Menu } from "lucide-react"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Button } from "@/components/ui/button"

export function NavMenu() {
  const pathname = usePathname()

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon" className="h-9 w-9 rounded-full">
          <Menu className="h-5 w-5" />
          <span className="sr-only">Navigation Menu</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem asChild>
          <Link href="/" className={pathname === "/" ? "font-bold" : ""}>
            Home
          </Link>
        </DropdownMenuItem>
        <DropdownMenuItem asChild>
          <Link href="/status" className={pathname === "/status" ? "font-bold" : ""}>
            Status
          </Link>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

