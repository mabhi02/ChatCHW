import { MedicalChat } from "@/components/MedicalChat"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { HeartPulse } from "lucide-react"

export default function HomePage() {
  return (
    <div className="container py-8 max-w-5xl mx-auto space-y-8">
      <Card className="border-none shadow-none bg-transparent">
        <CardHeader className="text-center space-y-4">
          <div className="mx-auto bg-primary/10 w-fit p-3 rounded-full">
            <HeartPulse className="w-8 h-8 text-primary" />
          </div>
          <div className="space-y-2">
            <CardTitle className="text-4xl font-bold">ChatCHW</CardTitle>
            <CardDescription className="text-lg">
              Next.JS AI-powered medical assistance for community health workers
            </CardDescription>
          </div>
        </CardHeader>
      </Card>

      <Card className="border shadow-md">
        <CardContent className="p-6">
          <MedicalChat />
        </CardContent>
      </Card>

      <div className="text-center text-sm text-muted-foreground">
        <p>AVM's Version</p>
      </div>
    </div>
  )
}